---
title: GPU Device Coverage Expansion — TDD Implementation Plan
spec: ./SPEC.md
research: ./research.md
status: draft
plan_version: 1
updated_at: 2026-08-07T00:00:00Z
tasks: 23
waves: 6
worktree: /home/user/Documents/workspace/catboost_rs/.claude/worktrees/gpu-string-param-oracle
branch: worktree-gpu-string-param-oracle
base_commit: 84dd2dc832bcf701cc9dd6d781cb36ac1fd0777e
---

# PLAN.md — GPU Device Coverage Expansion

This plan converts the LOCKED `SPEC.md` (GDC-01 … GDC-19, decisions D1–D4) into
independently executable Red→Green→Refactor tasks. It does **not** re-author the
specification's WHAT/WHY. Everything below is CodeGraph/`Read`-verified against
`base_commit` `84dd2dc` (the worktree's `crates/` tree is byte-identical to `main`
— `git diff --stat main -- crates/` is empty, so the repo-root CodeGraph index is
valid for this worktree).

---

## 0. Reading order for the implementing agent

1. §1 — verified corrections to SPEC.md's own "Unresolved" items. **Read before
   any task**; four of them change task shape, and three are hard blockers.
2. §2 — global rules (commands, lints, test conventions).
3. §3 — wave / dependency graph.
4. §4 — the 23 tasks.
5. §5 — coverage tables (spec → task, acceptance scenario → task).
6. §6 — blockers, assumptions, escalations.

---

## 1. CodeGraph verification results — every SPEC "Unresolved" item, resolved

Each item below was verified this session. `[V]` = verified fact with citation.
`[B]` = **blocker**: SPEC.md asserts something the code contradicts.

### V-1 — GDC-01: the `learning_folds_for_cycle` binding (RESOLVED)

`[V]` `crates/cb-train/src/boosting.rs:3832-3833`:

```rust
let learning_folds_for_cycle =
    crate::learning_fold_count(params.permutation_count, !ctr_candidates.is_empty());
```

It is a plain `usize` local in the same function (`train_inner`) as the call site
at `boosting.rs:4297`, bound ~465 lines earlier, never shadowed or moved. Threading
it is a one-token edit.

`[B-1]` **but**: `crates/cb-train/src/fold.rs:168` —
`learning_fold_count(pc, needed) = if needed { max(1, pc-1) } else { 1 }`, and
`needed` here is `!ctr_candidates.is_empty()`. Therefore **for a NON-CTR fit the
real value is always `1`**, whatever `permutation_count` is. SPEC GDC-01's first
Given/When/Then ("a fit with `permutation_count > 1` and NO CTR features … then
`fold_count` … not `1`") is unsatisfiable as written. The discriminating test must
use a fit with **CTR candidates present** and `permutation_count > 1`. T01 encodes
this correction.

### V-2 — GDC-03: the nonsym split-score path DOES have its own weighted-input sites

`[V]` `crates/cb-backend/src/kernels/nonsym_grow.rs` has **three** unweighted
der-gather sites, not one:

| line | site | what it feeds |
|---|---|---|
| 117-120 | `unsplit_score` | `LeafStats { sum_weighted_delta: sum_f64(&der_sub), … }` — the split-gain BASELINE |
| 149-150, 181-200 | `best_split_for_node` | the per-node histogram + left/right `LeafStats` — the SPLIT SCORE |
| 442-448 | leaf emission | `calc_average(sum_f64(&der_sub), sum_f64(&w_sub), scaled_l2)` — the LEAF VALUE |

`[V]` `crates/cb-backend/src/kernels/region_device.rs` mirrors this: split score via
`device_best_split_for_node(..., der1, weight, ...)` at lines 99 / 111 / 116, leaf
value at 163-170.

`[V]` **Both growers take `der1: &[f64]` and `weight: &[f64]` as plain host slice
parameters** (`nonsym_grow.rs:224-227`, `region_device.rs:57-59`), and every internal
use of `der1` is a *contribution sum* (never a residual carried forward). Therefore
the correct, minimal Green for GDC-03 **and** GDC-04 is a **single caller-side
substitution** at the two call sites in `GpuTrainSession::grow_one`
(`session.rs:1682-1684` region arm, `session.rs:1711-1714` nonsym arm): pass
`weighted_der1[i] = der1[i] * weight[i]` as the `der1` argument while `weight` stays
raw. This fixes leaf value, split score and unsplit baseline in one edit per arm and
touches **zero** lines inside `nonsym_grow.rs` / `region_device.rs`.

### V-3 — GDC-04: the Region grower file/line (RESOLVED)

`[V]` `crates/cb-backend/src/kernels/region_device.rs:163-170` (leaf value) and
`:99/:111/:116` (split score), called from `session.rs:1683-1694`. Covered by V-2's
caller-side fix.

### V-4 — GDC-02: the exact oblivious fix points (REFINED)

`[V]` `crates/cb-backend/src/gpu_runtime/session.rs:1933-1976` builds
`(score_der1_h, score_weight_h)` and calls `grow_oblivious_tree_resident`.
`[V]` inside `grow_oblivious_tree_resident` (`mod.rs:3981-4303`) the `der1_h`
parameter is used at **three** places, only one of which is the leaf reduce:

| line | use | must receive |
|---|---|---|
| 4025 | empty short-circuit — returned as the NEXT tree's `der1` | RAW der1 |
| 4207 | `launch_partition_split_packed_into` (object routing) | RAW der1 (numerically inert) |
| 4238 | `launch_partition_update_into` → leaf `calc_average` (4262) | **WEIGHTED der1** |

So GDC-02 must **not** simply swap the `der1_h` argument. Add a distinct
`leaf_der1_h: &Handle` parameter used only at 4238, and route the weighted product
into `score_der1_h` (line 4075 histogram) as well. `der1_h` stays raw, which
automatically satisfies GDC-02's third Given/When/Then (`launch_der_binary_resident`
at 4272 recomputes the residual from the resident approx and never reads the
parameter at all).

`[V]` the multiply primitive exists and is directly reusable:
`crates/cb-backend/src/kernels/bootstrap_device.rs:618-640` `fold_weights_resident`
launches `crate::kernels::vector_mul_kernel::launch::<f64, SelectedRuntime>` —
signature `(client, weight_h, sample_h, n) -> CbResult<Handle>`, pure elementwise
`a·b`, with a `#[cfg(feature = "wgpu")]` stub sibling. Call it as
`fold_weights_resident(&self.client, &der1_h, &self.weight_h, self.n)`. Do not fork.

### V-5 — GDC-13/D4: the blast radius is 10 sites, not 2 (CORRECTION)

`[V]` `grep -rn "grow_tree_on_device" crates/ --include=*.rs` (exhaustive):

| site | file:line | kind |
|---|---|---|
| trait default | `crates/cb-compute/src/runtime.rs:1376-1384` | signature + `let _ = (...)` |
| real override | `crates/cb-backend/src/gpu_backend.rs:310-335` | delegates to `session.grow_one` |
| production caller | `crates/cb-train/src/boosting.rs:4598` | the ONE call in `train_inner` |
| test wrapper | `crates/cb-train/tests/device_bootstrap_parity_test.rs:241,248` | `CountingGpu`-style, delegates to `self.inner` |
| test wrapper | `crates/cb-train/tests/device_poisson_bootstrap_test.rs:195,202` | idem |
| test wrapper | `crates/cb-train/tests/device_one_hot_parity_test.rs:193,199` | idem |
| test wrapper | `crates/cb-train/tests/bootstrap_dev_oracle_test.rs:261,267` | idem |
| test mock | `crates/cb-train/tests/device_seam_test.rs:69` | scripted-return mock |
| direct calls | `crates/cb-backend/src/gpu_backend_test.rs:189,200,228` | 3 call sites |

D4 remains correct in spirit (one *real* override), but the mechanical edit count is
**10**. T02 is sized for that, and its Green step lists every site.

`[V]` `CpuRefRuntime` in `device_nonsym_fit_test.rs` / `device_region_fit_test.rs`
overrides only `compute_gradients` and inherits the seam default — **unaffected**.

### V-6 — GDC-15: `body_tail_boundaries` crossing the seam (RESOLVED **and** re-typed)

`[V]` `crates/cb-train/src/fold.rs:105` — `pub fn body_tail_boundaries(n: usize,
multiplier: f64) -> Vec<usize>`; `:137` — `body_tail_segments(n, multiplier) ->
Vec<(usize, usize)>`. Pure functions of `(n, fold_len_multiplier)`. `Fold`
(`fold.rs:181-189`) holds `permutation: Vec<i32>` and `body_tail_boundaries:
Vec<usize>`.
`[V]` `crates/cb-backend/src/gpu_runtime/ordered.rs:98-168` — `OrderedTree` uses
`permutation: &[i32]` (it explicitly checks `doc_i < 0`), `body_finish: usize`,
`tail_finish: usize`.

**Correction to SPEC §4's typed contract**: `OrderedTreeArgs` must be

```rust
pub struct OrderedTreeArgs<'a> {
    pub permutation: &'a [i32],   // NOT [u32] — Fold::permutation and OrderedTree both use i32
    pub body_finish: usize,       // NOT &[u32] — a scalar per segment
    pub tail_finish: usize,       // NOT &[u32]
}
```

All primitives; **no `cb-train` type crosses the seam** → the T-10-04 feature-unification
landmine is not tripped. Verified answer to GDC-15's Unresolved item: **yes, safe**.

`[B-2]` **but**: the boundaries are a pure function of `(n, fold_len_multiplier)` —
they are **per-FIT constant**, not "per-tree dynamic, growing across iterations" as
SPEC GDC-13/GDC-15 state. The "growing" in `fold.cpp` is across *segments within one
fold*, not across boosting iterations. D4's signature change is therefore not
*required* by the data (the same values could ride `DeviceTrainConfig` at `begin`).
D4 is locked, so this plan **honours it** (per-tree carriage of per-fit-constant data
is correct, just not minimal) and flags the observation in §6 rather than
relitigating.

### V-7 — GDC-15/Track O: the device Ordered arm's real work is the SPLIT SEARCH (BLOCKER)

`[B-3]` `[V]` `crates/cb-train/src/boosting.rs:5572-5591` — the ONLY behavioural
difference between the CPU Ordered and Plain paths in `train_inner` is the structure
search: `greedy_tensor_search_oblivious_ordered(&matrix, &weighted_der1, &weights,
learning_perm, l2, fold_len_multiplier, depth, n)`. Leaf values then take the SAME
averaging-fold path as Plain ("`leaf_of` is object-order (Plain-identical) so the SAME
averaging-fold leaf-value path below applies").

`[V]` `crates/cb-train/src/tree.rs:2498-2548` `greedy_tensor_search_oblivious_ordered`
→ `select_level_ordered` (`:2427`) → `score_candidate_ordered` (`:2383-2418`): for
EVERY candidate `(feature, border)` at EVERY level it re-assigns leaves and sums
`l2_split_score` over every `(body_finish, tail_finish)` segment, each with its own
`scale_l2_reg(l2, body_sum_weight, body_finish)`.

`[V]` `grep -n "ordered_approx_delta_simple" crates/cb-train/src/boosting.rs
crates/cb-train/src/tree.rs crates/cb-train/src/lib.rs` → the function is defined
(`boosting.rs:736`) and re-exported (`lib.rs:104`) but **is never called anywhere in
the training loop**. It is an oracle/test primitive only.

**Consequence**: `cb_backend::gpu_runtime::ordered::accumulate_ordered_trajectory`
reproduces `ordered_approx_delta` — i.e. exactly the primitive the CPU trainer does
**not** use. Wiring it "keyed by the real `leaf_of`" (GDC-15 as written) would NOT
produce a fit matching the CPU/upstream Ordered model. The genuine device Ordered arm
needs a **per-segment ordered split score** (the `score_candidate_ordered` sum) on the
device path. This is materially larger than SPEC GDC-15 describes and has no existing
device precedent. T18 carries it and is marked **BLOCKED — decision required** (§6).

### V-8 — GDC-17: `ordered_boost_e2e` is NOT device-eligible (BLOCKER)

`[B-4]` `[V]` `crates/cb-oracle/fixtures/ordered_boost_e2e/config.json:2,16` pins
`"boost_from_average": true`; `[V]` `mean(y.npy) = 1.4695447467248968` over `n = 30`
⇒ the RMSE starting approx (`bias`) is ≈ 1.47, non-zero.
`[V]` `crates/cb-train/src/boosting.rs:4165-4170` — the **CR-01** clause
`&& bias == 0.0` is explicitly **out of scope** for this phase (SPEC §4 lists it among
the UNCHANGED clauses) because `GpuTrainSession::begin` seeds the resident approx to
zero.

⇒ SPEC GDC-17's "the ONE case with a directly reusable existing fixture — no new
fixture generation needed" is **false**. A NEW ordered fixture with
`boost_from_average=False` is required (T20).

### V-9 — GDC-12: every existing CTR e2e fixture is cat-ONLY and cannot reach the device (BLOCKER)

`[B-5]` `[V]` `ls crates/cb-oracle/fixtures/{plain_ctr,ctr_btmv_simple,ctr_buckets_simple,ctr_counter_simple,ctr_borders_multiprior,ctr_mixed_simple_vs_combo,tensor_ctr_e2e}`:
- `plain_ctr/` has **no `model.json` / `predictions.npy`** at all (it is a per-object
  stage fixture: `ctr_value.npy`, `permutation_fold0.npy`, …). SPEC's suggestion to
  reuse it as an e2e oracle is not possible.
- Every other CTR e2e fixture ships `X_cat.npy` only — **zero float columns**
  (deliberate: `ctr_btmv_simple/gen_fixtures.py` documents cat-only as the mitigation
  for upstream float-border nondeterminism).

`[V]` `crates/cb-train/src/boosting.rs:4208-4209`:
`device_n_float = matrix.n_features()` (float count only);
`device_n_features = device_n_float + one_hot_bins.len()`.
With `one_hot_max_size=1` both cat columns route to CTR, so
`device_n_features == 0` and `device_n_bins == 0`; `boosting.rs:4292`
(`device_host_eligible && device_n_bins > 0`) and `begin`'s own
`n_features == 0` decline both fire. **A cat-only CTR pool cannot reach the device
even after GDC-11.**

⇒ GDC-12 needs a NEW **mixed float+cat** CTR fixture (T13). Two additional pinned
constraints, both verified:
- `[V]` `ctr_covered` (`session.rs:130-144`) requires
  `col.borders.len() + 1 == n_bins` for **every** CTR column, where `n_bins` is the
  device histogram width derived from the FLOAT quantization. `[V]`
  `ctr_border_count_default() == 15` (`boosting.rs:573-575`) ⇒ 16 CTR buckets ⇒ the
  fixture's float `border_count` must be **15** (16 bins) or the CTR arm declines.
- `[V]` upstream float-border nondeterminism is already neutralised by the repo's own
  convention: `ordered_boost_e2e_oracle_test.rs:124-127` reads borders from the
  fixture's `model.json` via `model_json.float_feature_borders()` rather than
  re-deriving them.

### V-10 — GDC-09/GDC-11: type shapes (REFINED)

`[V]` `crates/cb-compute/src/runtime.rs:1048-1077`:
`DeviceCtrColumn { member_bins: Vec<Vec<u32>>, prior: f64, borders: Vec<f64> }`;
`DeviceCtrConfig { permutation: Vec<u32>, target_class: Vec<u32>, columns: Vec<DeviceCtrColumn> }`.
SPEC §4 says `target_class: Vec<u8>` — the real type is `Vec<u32>`. `DeviceCtrAveraging`
must mirror the real types.
`[V]` `build_ctr_cindex_columns` (`session.rs:157-188`) already takes
`ctr: &DeviceCtrConfig` and returns `Vec<Vec<u32>>` — GDC-09's "call the same kernel a
second time" is a **parameter refactor** (take `permutation: &[u32]`, `target_class:
&[u32]`, `columns: &[DeviceCtrColumn]` instead of the whole config) plus one extra
call, not new kernel code.

### V-11 — Ordered × CTR is ALREADY mutually excluded backend-side (defence in depth)

`[V]` `map_ordered_coverage` (`session.rs:543-566`) already requires
`config.ctr.is_none()`. `[V]` `ctr_is_covered` (`session.rs:1186-1195`) is only
reachable on the Plain branch. So even without a new `boosting.rs` guard, an
Ordered+CTR fit would decline at `begin` → `device_active == false` → correct CPU
path. GDC-16's explicit host-side guard is still required by SPEC (fail-fast, and it
keeps the decision at the single D-10-01 gate); GDC-19.1's negative test will pass
against **either** layer, so T23's assertion must target the observable
(`device_active == false` via a counting runtime), not the specific clause.

### V-12 — file-ownership overlap between the three tracks (SPEC's parallelism claim, corrected)

| file | Track W | Track C | Track O |
|---|---|---|---|
| `cb-backend/src/gpu_runtime/session.rs` `grow_one` (1620-2020) | **T04/T05/T06** | — | T19 |
| `cb-backend/src/gpu_runtime/session.rs` `begin` (949-1230) | — | **T11/T14-side** | **T16** |
| `cb-backend/src/gpu_runtime/session.rs` helpers (120-190) | — | **T11** | — |
| `cb-backend/src/gpu_runtime/mod.rs` `grow_oblivious_tree_resident` | **T04** | T12 (leaf reduce) | — |
| `cb-backend/src/kernels/nonsym_grow.rs` / `region_device.rs` | untouched (V-2) | — | — |
| `cb-compute/src/runtime.rs` | — | **T10** | **T02** |
| `cb-train/src/boosting.rs` `device_host_eligible` | **T07** | **T14** | **T21** |

⇒ SPEC's "mostly-disjoint files" claim is **only partly true**. Verified reality:
- Track W's three kernel tasks (T04/T05/T06) all edit `GpuTrainSession::grow_one` —
  serialise them under one owner.
- Track C's `begin` edits (T11) and Track O's `begin` edit (T16) touch the same
  function ~250 lines apart — low but non-zero conflict; serialise T16 before T11 or
  coordinate.
- T02 and T10 both edit `cb-compute/src/runtime.rs` (different structs/impls) —
  parallel-safe with a rebase.
- T07 / T14 / T21 all edit the **same expression** (`device_host_eligible`,
  `boosting.rs:4083-4176`) — **strictly serialise**: T07 → T14 → T21.
- T12 (CTR leaf gather) reaches into `grow_oblivious_tree_resident`'s leaf-reduce
  region (`mod.rs:4227-4264`), the same region T04 edits — serialise T04 before T12.

---

## 2. Global rules for every task

### 2.1 Commands (exact; deviations are documented traps)

| purpose | command |
|---|---|
| backend build-verify (no GPU needed) | `cargo check -p cb-backend --no-default-features --features rocm` |
| backend self-oracles (real device) | `cargo test -p cb-backend --no-default-features --features rocm` |
| backend self-oracle, filtered | `cargo test -p cb-backend --no-default-features --features rocm -- <filter>` |
| compute-crate types | `cargo check -p cb-compute` |
| new cb-train device test, build-verify | `cargo check -p cb-train --no-default-features --features rocm --test <name>` |
| new cb-train device test, real run | `cargo test -p cb-train --no-default-features --features rocm --test <name>` |
| CPU regression (default features) | `cargo test -p cb-train --test <name>` |
| workspace non-device regression | `cargo test --workspace` |

**Traps (research.md pitfalls 5 & 6, both reproduced with real `cargo` output):**
- NEVER `--features rocm` without `--no-default-features` — `cpu` stays on,
  `SelectedRuntime` resolves to `cubecl-cpu`, and device tests either SKIP silently or
  compare cpu-vs-cpu and report a **false pass**.
- NEVER a package-wide `cargo test -p cb-train --no-default-features --features rocm`
  without `--test <name>` — ~52 test files `use cb_backend::CpuBackend;` at module
  scope and fail `E0432`, blocking the whole run.

### 2.2 New-test conventions (mandatory, verified against `device_nonsym_fit_test.rs`)

Every new `crates/cb-train/tests/device_*.rs` file MUST:
1. carry `#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing, clippy::float_cmp)]`;
2. put every device-touching item inside `#[cfg(any(feature = "rocm", feature = "cuda"))] mod device { … }`;
3. **never** `use cb_backend::CpuBackend` (it is not compiled under rocm/cuda and
   breaks `--no-default-features`); use a local `struct CpuRefRuntime;` that overrides
   only `compute_gradients` and inherits every seam default;
4. mark shared helpers `#[cfg_attr(not(any(feature = "rocm", feature = "cuda")), allow(dead_code))]`;
5. **SKIP by printing and returning** on cpu/wgpu (`eprintln!("SKIP …: needs rocm/cuda")`)
   — never `panic!`, never `#[ignore]` (the WR-01 anti-false-pass convention);
6. assert the **right arm actually fired** (tree-kind routing, or a `CountingGpu`
   wrapper counting `grow_tree_on_device` calls — precedent
   `bootstrap_dev_oracle_test.rs:261-267`), so a silent CPU fallback cannot pass.

### 2.3 Source rules

- Source/test separation is mandatory: no `#[cfg(test)] mod tests` in a production
  file. `cb-backend` self-oracles live in sibling `*_test.rs` files mounted as child
  modules (existing pattern).
- No `unwrap` / `expect` / `panic` / raw indexing anywhere in `crates/cb-backend/src/**`
  (workspace lints + D-13). Use `.get(..).copied().unwrap_or(..)` / `let … else`.
- Any host reduction routes through `cb_core::sum_f64` (D-07/D-08).
- Do Not Hand-Roll: `vector_mul_kernel`/`fold_weights_resident` (elementwise multiply),
  `launch_ordered_ctr_resident` + `binarize_ctr_column_resident` (online CTR),
  `launch_apply_leaf_delta_into` (leaf-delta apply), `cb_core::sum_f64`.
- Preserve D-04 (byte-unchanged CPU path for non-admitted configs), D-05 (only O(1)
  per-level crossings), D-10-01 (one per-fit all-or-nothing decision).
- D1: **nothing merges until every task in this plan is green.** Wave order below is
  for dependency correctness only.

### 2.4 Documented residual (SPEC §9, carried forward)

The fixed-point histogram's `|Σ| < 2^33` precondition (`kernels.rs:2318-2335`) has no
enforced runtime guard. This phase does not close it. Every new weighted fixture must
stay comfortably under `n · max(weight) · max(|der1|) < 2^33` **by construction** and
document the margin at the generator call site. Report this in the completion summary.

---

## 3. Waves and dependency graph

```text
WAVE 0  (parallel: A, B, C are disjoint)
  A: T01 (GDC-01)                                   boosting.rs:4297
  B: T02 (GDC-13)                                   cb-compute runtime.rs + 9 sites
  C: T03 (GDC-06/W fixtures)                        cb-oracle generator
  D: T10 (GDC-09a types)                            cb-compute runtime.rs (DeviceCtrConfig)
  E: T13 (GDC-06/C fixture)   [unblocks after §6 Q2]
  F: T20 (GDC-06/O fixture)   [unblocks after §6 Q1]

WAVE 1  (device kernels; W is a serial chain inside session.rs::grow_one)
  T04 (GDC-02) -> T05 (GDC-03) -> T06 (GDC-04)        [Track W]
  T16 (GDC-14) -> T17 (GDC-18)                        [Track O]   needs T02
  T11 (GDC-09b)                                       [Track C]   needs T10; after T16
  T18 (GDC-15a) BLOCKED                               [Track O]   needs T02,T16

WAVE 2
  T12 (GDC-10)                                        [Track C]   needs T11, T04
  T19 (GDC-15b)                                       [Track O]   needs T18

WAVE 3  (the gate — STRICTLY SERIAL, same expression)
  T07 (GDC-05) -> T14 (GDC-11) -> T21 (GDC-16)
     T07 needs T04,T05,T06
     T14 needs T01,T11,T12,T07
     T21 needs T18,T19,T14

WAVE 4  (e2e oracles; parallel with each other)
  T08 (GDC-07)  needs T03,T04,T07
  T09 (GDC-08)  needs T03,T05,T06,T07
  T15 (GDC-12)  needs T13,T14
  T22 (GDC-17)  needs T20,T21

WAVE 5
  T23 (GDC-19)  needs T07,T14,T21
```

ASCII dependency edges (acyclic, verified):

```text
T01 ─────────────────────────────────────► T14
T02 ─► T16 ─► T17
   └──► T18 ─► T19 ─► T21
T03 ─► T08, T09
T04 ─► T05 ─► T06 ─► T07 ─► T14 ─► T21 ─► T22
T04 ─────────────► T12
T10 ─► T11 ─► T12 ─► T14
T13 ─► T15;   T14 ─► T15
T16 ─► T11 (file-order coordination only)
T20 ─► T22
T07, T14, T21 ─► T23
```

---

## 4. Tasks

---

### T01 — Thread the real `learning_folds_for_cycle` into `begin_device_training`

- **Specs**: GDC-01
- **Depends on**: none — **WAVE 0**, parallelizable with everything.
- **Parallel**: YES. Sole edit is one argument at `boosting.rs:4297`; no other task
  touches that line (T07/T14/T21 edit `device_host_eligible` at 4083-4176).

**Goal / observable completion**: `begin_device_training`'s `fold_count` argument
carries the real `learning_folds_for_cycle`. Observable: a CTR-candidate fit with
`permutation_count > 1` reaching the device gate hands `fold_count > 1` to the
backend, which then declines (`ctr_covered`'s `fold_count != 1`, `session.rs:130-144`,
and `begin`'s `fold_count != 1` at `session.rs:992`).

**Files / symbols (verified)**
- Modify: `crates/cb-train/src/boosting.rs:4297` — `/* fold_count = */ 1,` →
  `learning_folds_for_cycle,` (binding at `:3832-3833`, same fn scope).
- Test: **new** `crates/cb-train/tests/device_fold_count_gate_test.rs`.

**Red**
- Test name: `ctr_multi_permutation_fit_does_not_commit_to_device`.
- Setup: a fit with ≥1 categorical column routed to CTR (`one_hot_max_size = 1`), at
  least one float column, `permutation_count = 4`, otherwise device-eligible params
  (RMSE/Logloss, `bootstrap_type=No`, `random_strength=0`, `boost_from_average=false`,
  `LeafMethod::Gradient`, `SymmetricTree`). Drive it through `cb_train::train_cat`
  (`boosting.rs:2614`) with a `CountingGpu`-style runtime that records the
  `fold_count` it was handed by `begin_device_training`.
- Expected failure: the recorded `fold_count` is `1`, but the assertion demands
  `learning_fold_count(4, true) == 3`.
- **Correction (V-1)**: do **not** write the "no CTR features" variant SPEC GDC-01
  lists first — with `ctr_candidates.is_empty()` the real value is `1` and the test
  would be vacuous. Add instead a second, complementary case
  `plain_fit_still_passes_fold_count_one` asserting `fold_count == 1` for a
  non-CTR / `permutation_count = 1` fit (the D-04 byte-unchanged guard).
- Run: `cargo check -p cb-train --no-default-features --features rocm --test device_fold_count_gate_test`
  then `cargo test -p cb-train --test device_fold_count_gate_test`.

**Green**
- Replace the literal with `learning_folds_for_cycle`. Nothing else.
- Do NOT relax any CTR clause here (GDC-11/T14 owns that).

**Refactor**
- Add a one-line comment at 4297 citing GDC-01 and the `ctr_covered` `fold_count != 1`
  contract. No behavioural change.

**Verify**
- `cargo test -p cb-train --test device_fold_count_gate_test`
- Regression: `cargo test -p cb-train --test device_seam_test --test device_bootstrap_parity_test --test device_one_hot_parity_test`
- `cargo check -p cb-train --no-default-features --features rocm --test device_fold_count_gate_test`
- Confirm: no currently-green device e2e test changes behaviour (all pin
  `permutation_count` default with no CTR ⇒ real value stays `1`).

**Completion evidence**: both test functions green; the recorded `fold_count` is `3`
for the CTR case and `1` for the plain case.

**Risks**: the CTR fit must not already be excluded for an unrelated reason before it
reaches `begin_device_training` — at this point in the plan the CTR clause at
`boosting.rs:4085-4086` still excludes it, so `begin_device_training` is **not
called**. Therefore the Red test must assert on `device_host_eligible`'s *input*
rather than the backend call. **Mitigation**: assert via a unit-visible helper — extract
the `fold_count` expression into a `#[must_use] fn device_fold_count(...)` (or assert
through the T14 path later). If extraction is undesirable, defer the positive
assertion to T14's own test and keep T01's Red as the `plain_fit_still_passes_fold_count_one`
regression plus a compile-level proof. **Decide at execution time and record the choice
in the task's completion note.**

---

### T02 — Extend `Runtime::grow_tree_on_device` with `ordered: Option<&OrderedTreeArgs>`

- **Specs**: GDC-13
- **Depends on**: none — **WAVE 0**.
- **Parallel**: YES with T01/T03/T13/T20. Shares `cb-compute/src/runtime.rs` with T10
  (different items) — rebase-safe, but coordinate.

**Goal / observable completion**: the workspace compiles with the new parameter
threaded through all 10 verified sites; every pre-existing test stays green; no
behavioural change for any non-Ordered fit.

**Files / symbols (all verified, V-5)**
- Modify: `crates/cb-compute/src/runtime.rs:1376-1384` — the trait method + its
  `let _ = (approx, target, sample, ordered);` default.
- Create (same file): `pub struct OrderedTreeArgs<'a>` per **V-6**'s corrected types
  (`permutation: &'a [i32]`, `body_finish: usize`, `tail_finish: usize`), plain host
  type, no `cubecl` import (T-10-04).
- Modify: `crates/cb-backend/src/gpu_backend.rs:310-335` — add the parameter, bind
  it (`let _ = ordered;`) this task; T19 makes it load-bearing.
- Modify: `crates/cb-train/src/boosting.rs:4598` — pass `None`.
- Modify (test wrappers, pass through to `self.inner`):
  `crates/cb-train/tests/device_bootstrap_parity_test.rs:241,248`,
  `device_poisson_bootstrap_test.rs:195,202`,
  `device_one_hot_parity_test.rs:193,199`,
  `bootstrap_dev_oracle_test.rs:261,267`,
  `device_seam_test.rs:69`.
- Modify (direct calls): `crates/cb-backend/src/gpu_backend_test.rs:189,200,228` —
  add `None`.

**Red**
- This spec's risk is a **compile-time** regression, so the Red is a build failure by
  construction: change the trait signature FIRST and run
  `cargo check -p cb-compute && cargo check -p cb-backend --no-default-features --features rocm`.
- Expected failure: `E0050`/`E0061` (method takes N parameters but N+1 supplied) at
  `gpu_backend.rs:310` and each test wrapper — enumerate them and confirm the count is
  exactly the 10 sites in V-5. A site NOT in that list means the blast radius was
  mis-measured; stop and re-run `grep -rn "grow_tree_on_device" crates/ --include=*.rs`.
- Additionally add `crates/cb-compute/src/runtime_ordered_args_test.rs` (new sibling
  test file, source/test separation) with `ordered_tree_args_is_plain_host_type`
  constructing an `OrderedTreeArgs` from `&[i32]`/`usize` literals and asserting field
  round-trip — fails to compile before the struct exists.

**Green**
- Add the struct and the parameter; thread `None` everywhere except (later) T19.
- The trait default stays `Ok(None)` unconditionally — byte-unchanged.

**Refactor**
- Doc-comment the new parameter with the D4 rationale AND the V-6 note that the values
  are per-fit constants carried per tree.

**Verify**
- `cargo check -p cb-compute`
- `cargo check -p cb-backend --no-default-features --features rocm`
- `cargo test --workspace` (default features) — every pre-existing test green.
- `cargo check -p cb-train --no-default-features --features rocm --test device_nonsym_fit_test`
- `cargo check -p cb-train --no-default-features --features rocm --test bootstrap_dev_oracle_test`
- Confirm: `git diff --stat` shows exactly the 10 sites + the new struct + the new test.

**Completion evidence**: clean workspace build, `cargo test --workspace` green, diff
limited to the enumerated sites.

**Risks**: a wrapper that forgets to forward `ordered` would silently drop ordered
state later. **Guardrail**: every test wrapper must forward the parameter verbatim
(`self.inner.grow_tree_on_device(approx, target, sample, ordered)`), never `None`.

---

### T03 — Weighted-training oracle fixtures (SymmetricTree + one non-symmetric)

- **Specs**: GDC-06 (weighted scenarios)
- **Depends on**: none — **WAVE 0**. Requires a local `catboost==1.2.10` install
  (generation is OFFLINE; CI only reads the committed artifacts, D-12).
- **Parallel**: YES — touches only `crates/cb-oracle/`.

**Goal / observable completion**: two frozen fixture directories exist, each loadable
by a Rust smoke test, produced by the installed upstream package.

**Scenario count (SPEC GDC-06 left this to the planner — DECIDED)**: **two**
scenarios, not four.
- `weighted_device_sym/` — RMSE, `grow_policy=SymmetricTree`, depth 3.
- `weighted_device_nonsym/` — RMSE, `grow_policy=Depthwise`, depth 3.
Rationale: Depthwise / Lossguide / Region share the *same* verified fix point (V-2:
one caller-side substitution per arm) and the same `calc_average` contract; a
per-policy upstream fixture would triple generation cost for zero additional failure
isolation. T09 parameterises the *device* run over all three policies against the
CPU reference at ε=1e-4 and against `weighted_device_nonsym/predictions.npy` at ≤1e-5
for Depthwise. **If T09's Lossguide/Region device-vs-CPU comparison diverges, add the
missing upstream scenario then — do not pre-generate.**

**Files / symbols**
- Create: `crates/cb-oracle/fixtures/weighted_device_sym/{gen_fixtures.py,config.json,X.npy,y.npy,weights.npy,model.json,predictions.npy}`
- Create: `crates/cb-oracle/fixtures/weighted_device_nonsym/{…same…}`
- (Follow the per-fixture-directory generator convention verified in
  `crates/cb-oracle/fixtures/ctr_btmv_simple/gen_fixtures.py`, which is how recent
  fixtures are authored; the central `crates/cb-oracle/generator/gen_fixtures.py`
  hosts older scenarios.)

**Pinned recipe (every value load-bearing)**
```
catboost==1.2.10, numpy.random.RandomState(0), thread_count=1
n_rows = 64, n_float_features = 2, border_count = 15   # -> 16 bins
loss_function = "RMSE"
iterations = 3, depth = 3, learning_rate = 0.3, l2_leaf_reg = 3.0
boosting_type = "Plain", bootstrap_type = "No", random_strength = 0
boost_from_average = False          # bias == 0.0 (CR-01 stays unrelaxed)
leaf_estimation_method = "Gradient", leaf_estimation_iterations = 1
score_function = "L2"
grow_policy = "SymmetricTree" | "Depthwise"
sample_weight = non-uniform, w[i] in {0.5, 1.0, 2.0, 3.0} (cycled)
```
- **Overflow margin (SPEC §9, mandatory comment in the generator)**:
  `n · max(w) · max(|der1|) = 64 · 3.0 · max|y| ≪ 2^33 ≈ 8.6e9`. Bound `|y| ≤ 10`
  ⇒ worst case `1920`, margin `> 4.4e6×`. Record this arithmetic verbatim.
- Anti-false-pass guard in the generator: assert the trained model has ≥1 non-trivial
  split and that predictions differ from an unweighted-trained model by > 1e-6
  (otherwise the fixture cannot discriminate the weighted formula).

**Red**
- Test: `crates/cb-oracle/tests/weighted_device_fixture_smoke_test.rs`
  (or extend the existing fixture-smoke pattern), function
  `weighted_device_fixtures_load`.
- Setup: load `X.npy`, `y.npy`, `weights.npy`, `predictions.npy` via the existing
  `load_f64_vec` / `load_model_json` helpers; assert shapes, assert
  `weights.iter().any(|&w| w != 1.0)` (proves non-uniformity), assert
  `model_json.float_feature_borders()` yields 2 columns of 15 borders.
- Expected failure: fixture directories do not exist ⇒ load error.
- Run: `cargo test -p cb-oracle --test weighted_device_fixture_smoke_test`

**Green**
- Run each `gen_fixtures.py` OFFLINE against the installed `catboost==1.2.10`; commit
  the artifacts. Mark `"note": "FROZEN … NEVER regenerated in CI"` in `config.json`,
  matching every other fixture.

**Refactor**
- Factor the two generators' shared body into one helper inside the fixture dir if the
  duplication exceeds ~30 lines; do NOT move it into the central generator (recent
  convention is per-directory).

**Verify**
- `cargo test -p cb-oracle --test weighted_device_fixture_smoke_test`
- Confirm: `git status` shows only new files under `crates/cb-oracle/fixtures/weighted_device_*`.

**Completion evidence**: both directories committed; smoke test green; the overflow
margin comment present in both generators.

**Risks**: upstream quantization nondeterminism on float columns. **Mitigation**: the
Rust side reads borders from the committed `model.json` (verified precedent
`ordered_boost_e2e_oracle_test.rs:124-127`), so a re-run's different borders cannot
silently invalidate the gate — but the artifacts are frozen and must never be
regenerated.

---

### T04 — Weighted-der channel in the oblivious resident grow (SymmetricTree)

- **Specs**: GDC-02
- **Depends on**: none technically, but **serialise T04 → T05 → T06** (same function).
- **Parallel**: NO within Track W. YES vs Tracks C/O except T12 (see V-12).

**Goal / observable completion**: with a non-uniform weight vector, the device
oblivious grow's split histogram AND leaf reduce both consume `der1 ⊙ weight`; leaf
values equal `Σ(w·der1) / (Σw + l2)`. With uniform weights the result is numerically
identical to today.

**Files / symbols (verified, V-4)**
- Modify: `crates/cb-backend/src/gpu_runtime/mod.rs:3981-4013` — add a
  `leaf_der1_h: &Handle` parameter (place it immediately after `der1_h`).
- Modify: `crates/cb-backend/src/gpu_runtime/mod.rs:4236-4245` — feed `leaf_der1_h`
  (not `der1_h`) to `launch_partition_update_into`.
- Leave `mod.rs:4025` (empty short-circuit carry-forward) and `mod.rs:4207`
  (`launch_partition_split_packed_into` routing) reading the RAW `der1_h`.
- Modify: `crates/cb-backend/src/gpu_runtime/session.rs:1933-1976`:
  ```
  let weights_uniform: bool = <precomputed once at begin, stored on the session>;
  let weighted_der1_h = if weights_uniform { der1_h.clone() }
                        else { fold_weights_resident(&self.client, &der1_h, &self.weight_h, self.n)? };
  let (score_der1_h, score_weight_h) = match effective_sample_h.as_ref() {
      Some(s) => (fold_weights_resident(&self.client, &weighted_der1_h, s, self.n)?,
                  fold_weights_resident(&self.client, &self.weight_h, s, self.n)?),
      None => (weighted_der1_h.clone(), self.weight_h.clone()),
  };
  // pass &der1_h (raw), &weighted_der1_h (leaf), &score_der1_h, &score_weight_h
  ```
- Modify: `crates/cb-backend/src/gpu_runtime/session.rs` `begin` / the session struct —
  add a `weights_uniform: bool` field computed once from the `weight` slice
  (`weight.iter().all(|&w| w == 1.0)`), so the uniform path launches **zero** extra
  kernels (D-04 byte-and-perf-unchanged).
- Reuse only: `crates/cb-backend/src/kernels/bootstrap_device.rs:618-640`
  `fold_weights_resident`. Do not fork, do not author a new kernel.

**Red**
- Extend `crates/cb-backend/src/gpu_runtime/session_residency.rs`'s existing
  `weight_mod5(n)` (`:37`) weighted scenario with a NEW, tight assertion — SPEC GDC-02
  explicitly says to correct the existing test rather than only add a file. New test
  function: `resident_leaf_values_match_weighted_calc_average`.
- Setup: `n = 64`, `weight = weight_mod5(n)` (non-uniform, verified helper), a fixed
  2-feature 16-bin cindex, RMSE der.
- Assertion: for every leaf, device `leaf_values[leaf]` equals the host reference
  `calc_average(sum_f64(&w·der1 over leaf), sum_f64(&w over leaf), scaled_l2)` at
  **ε = 1e-4** — strictly tighter than the file's existing generous
  `LEAF_BOUND = 1e-3` (`session_residency.rs:26`), which is exactly why it passes
  today despite the bug.
- Expected failure: the device returns `Σder1 / (Σw + l2)`; with `weight_mod5`
  (weights in `{1,2,3,4,5}`) the numerator is off by a factor of order the mean weight
  ⇒ divergence far above 1e-4.
- Run: `cargo test -p cb-backend --no-default-features --features rocm -- resident_leaf_values_match_weighted_calc_average`

**Green**
- Implement exactly the substitution above. Nothing else.
- Do NOT touch `nonsym_grow.rs` / `region_device.rs` (T05/T06).
- Do NOT relax any `boosting.rs` clause (T07).

**Refactor**
- Consider a thin `weighted_der1_resident(client, der1_h, weight_h, n)` wrapper in
  `bootstrap_device.rs` delegating to `fold_weights_resident`, so the call site reads
  in weighted-der terms rather than bootstrap terms. Cosmetic only; keep the
  `#[cfg(feature = "wgpu")]` stub sibling in sync.
- Document the `|Σ| < 2^33` precondition at the new call site (SPEC §2's explicit
  "document the bound at the new call site" requirement).

**Verify**
- `cargo test -p cb-backend --no-default-features --features rocm`
  (all existing self-oracles: `ordered_test.rs`, `ctr_device_test.rs`,
  `session_depth_gt1_test.rs`, `session_residency.rs`, `nonsym_grow_test.rs`,
  `region_device_test.rs` — every one must stay green).
- `cargo check -p cb-backend --no-default-features --features rocm`
- Confirm: the uniform-weight path takes the `weights_uniform` short-circuit
  (add a debug assertion or a `CB_GPU_PROF` line proving zero extra launches).

**Completion evidence**: the new tight assertion green on real rocm; every
pre-existing cb-backend self-oracle green.

**Risks**: passing weighted der1 into `launch_partition_split_packed_into` (4207) or
the empty short-circuit (4025) would corrupt routing / the carried residual.
**Guardrail**: the Green step above keeps those two on the RAW handle by construction;
re-read the diff at `mod.rs:4025` and `:4207` before declaring done.

---

### T05 — Weighted-der channel for `grow_nonsym_tree` (Depthwise / Lossguide)

- **Specs**: GDC-03 (D3)
- **Depends on**: T04 (same function, `GpuTrainSession::grow_one`).
- **Parallel**: NO vs T04/T06/T19.

**Goal / observable completion**: with non-uniform weights and
`grow_policy ∈ {Depthwise, Lossguide}`, leaf values, split scores AND the unsplit
baseline all use `Σ(w·der1)`, matching the CPU `leaf_wise_grower` reference at ε=1e-4.

**Files / symbols (verified, V-2)**
- Modify: `crates/cb-backend/src/gpu_runtime/session.rs:1711-1725` — the nonsym arm
  of `grow_one`:
  ```
  let der1 = host_der1(ns.der_kernel, approx, target);
  let weighted: Vec<f64> = der1.iter().zip(ns.weight.iter()).map(|(&d, &w)| d * w).collect();
  return grow_nonsym_tree(ns.policy, &weighted, &ns.weight, &ns.bins, …);
  ```
  (a raw elementwise map is correct here — this is a *transform*, not a reduction, so
  `cb_core::sum_f64` does not apply; every downstream **reduction** inside
  `nonsym_grow.rs` already routes through `sum_f64`, verified at `:120, :199, :200, :448`).
- Do **NOT** edit `crates/cb-backend/src/kernels/nonsym_grow.rs` — V-2 proves the
  caller-side substitution covers all three internal sites (117-120 baseline,
  149-150/181-200 split score, 442-448 leaf value).

**Red**
- Extend `crates/cb-backend/src/kernels/nonsym_grow_test.rs` (existing functions
  verified: `depthwise_matches_cpu_leaf_wise_l2/_cosine`,
  `lossguide_matches_cpu_leaf_wise_l2/_cosine`, helper `assert_matches(policy,
  score_fn, label)` at `:339`, fixture at `:327`).
- New: parameterise `fixture(n, n_features, n_bins)` / `assert_matches` with a
  `weight: &[f64]` argument and add four weighted cases:
  `depthwise_weighted_matches_cpu_leaf_wise_{l2,cosine}`,
  `lossguide_weighted_matches_cpu_leaf_wise_{l2,cosine}`.
- Setup: the same fixture with `weight[i] = 1.0 + (i % 4) as f64` (non-uniform, and
  well inside the 2^33 bound), and a host reference `cpu_leaf_wise` (`:147`) fed the
  SAME weighted der.
- Expected failure: unweighted device leaf values / split choices diverge from the
  weighted CPU reference — note the split STRUCTURE may also differ, so assert leaf
  values **and** the chosen `(feature, bin)` per node.
- Run: `cargo test -p cb-backend --no-default-features --features rocm -- nonsym_grow`

**Green**
- The two-line caller-side substitution above. Nothing inside the kernel module.

**Refactor**
- Hoist the `der1 ⊙ weight` map into one private helper in `session.rs`
  (`fn host_weighted_der1(der1: &[f64], weight: &[f64]) -> Vec<f64>`) shared by T05 and
  T06 — do this in T06's Refactor, once both call sites exist.

**Verify**
- `cargo test -p cb-backend --no-default-features --features rocm -- nonsym_grow`
- `cargo test -p cb-backend --no-default-features --features rocm` (full self-oracle set)
- Confirm: the four UNWEIGHTED pre-existing cases stay green byte-unchanged.

**Completion evidence**: 4 new weighted cases green; 4 pre-existing cases green.

**Risks**: the CPU reference `cpu_leaf_wise` must be fed the weighted der too, or the
test asserts the wrong thing. **Guardrail**: assert first that with UNIFORM weights the
weighted and unweighted paths produce identical output (a self-check inside the test).

---

### T06 — Weighted-der channel for the Region grower

- **Specs**: GDC-04 (D3)
- **Depends on**: T05 (same function).
- **Parallel**: NO vs T04/T05/T19.

**Goal / observable completion**: with non-uniform weights and `grow_policy=Region`,
the device Region path's leaf values and per-level split choices use `Σ(w·der1)`,
matching the CPU `region_grower` reference at ε=1e-4.

**Files / symbols (verified, V-3)**
- Modify: `crates/cb-backend/src/gpu_runtime/session.rs:1682-1694` — the region arm:
  compute `weighted` from `der1` and `rg.weight`, pass it as `grow_region_tree`'s first
  argument, keep `&rg.weight` as the second.
- Do **NOT** edit `crates/cb-backend/src/kernels/region_device.rs` (leaf value at
  `:163-170`, split scores at `:99/:111/:116` are all covered).

**Red**
- Extend `crates/cb-backend/src/kernels/region_device_test.rs` (existing function
  verified: `region_device_reproduces_frozen_cpu_region_path` at `:63`).
- New: `region_device_weighted_reproduces_cpu_region_path` with
  `weight[i] = 1.0 + (i % 4) as f64`.
- Expected failure: unweighted leaf values (and possibly a different `region_path`)
  vs the weighted CPU reference.
- Run: `cargo test -p cb-backend --no-default-features --features rocm -- region_device`

**Green**
- The caller-side substitution.

**Refactor**
- Extract the shared `host_weighted_der1` helper now (T05's deferred refactor) and use
  it from both arms. Doc-comment it with the CPU parity target
  (`boosting.rs:5193-5205` `weighted_der1`) and the `|Σ| < 2^33` note.

**Verify**
- `cargo test -p cb-backend --no-default-features --features rocm -- region_device`
- `cargo test -p cb-backend --no-default-features --features rocm`
- `cargo check -p cb-backend --no-default-features --features rocm`

**Completion evidence**: new weighted Region case green; the frozen unweighted case
green byte-unchanged.

---

### T07 — Remove `device_host_eligible`'s weight-uniformity clause

- **Specs**: GDC-05
- **Depends on**: T04, T05, T06 (D-04: never admit before the math is right).
- **Parallel**: NO — first of the strictly-serial gate chain T07 → T14 → T21.

**Goal / observable completion**: a non-uniform-weight fit on any of the four covered
grow policies is no longer forced ineligible by weights alone, and `device_active`
observably becomes `true` for a configuration that was `false` before.

**Files / symbols (verified)**
- Modify: `crates/cb-train/src/boosting.rs:4155-4164` — delete the
  `&& weights.iter().all(|&w| w == 1.0)` clause and its WR-03 comment block; replace
  the comment with a GDC-05 note pointing at T04/T05/T06's fix points and the
  unrelaxed `bias == 0.0` (CR-01) sibling.
- Per **D3**, the clause is REMOVED, not narrowed to `SymmetricTree`.

**Red**
- Test: `crates/cb-train/tests/device_weight_gate_test.rs`, function
  `non_uniform_weights_commit_to_device`.
- Setup: the `device_nonsym_fit_test.rs` clear-margin fixture + `nonsym_params(...)`
  shape, but with `weights = [1.0, 2.0, 1.0, 3.0, …]`; run
  `cb_train::train(&CountingGpu::new(GpuBackend::default()), …, &weights, …)` for each
  of `SymmetricTree`, `Depthwise`, `Lossguide`, `Region`.
- Assertion: the counting wrapper recorded `iterations` calls to
  `grow_tree_on_device` (i.e. `device_active == true`) — the research.md pitfall-3
  anti-false-pass shape.
- Expected failure: **zero** device grow calls (the clause forces CPU).
- Run: `cargo check -p cb-train --no-default-features --features rocm --test device_weight_gate_test`,
  then `cargo test -p cb-train --no-default-features --features rocm --test device_weight_gate_test`.

**Green**
- Delete the clause.

**Refactor**
- Re-flow the surviving comment block; verify no other clause was disturbed by reading
  the full 4083-4176 expression back.

**Verify**
- `cargo test -p cb-train --no-default-features --features rocm --test device_weight_gate_test`
- `cargo test -p cb-train --no-default-features --features rocm --test device_nonsym_fit_test`
- `cargo test -p cb-train --no-default-features --features rocm --test device_region_fit_test`
- `cargo test -p cb-train --no-default-features --features rocm --test bootstrap_dev_oracle_test`
- `cargo test --workspace` (CPU regression — a weighted fit that USED to run on CPU now
  runs on CPU still under default features, since `GpuBackend` is not compiled there).
- Confirm: **acceptance scenario 1** — every uniform-weight fit is byte-unchanged.

**Completion evidence**: 4 policies × device-active assertion green; no pre-existing
device test regressed.

**Risks**: admitting weights also admits weights × bootstrap sampling
(SymmetricTree-only per the untouched clause at `:4118-4125`) — the score channel
composition `fold(weighted_der1, sample)` in T04 already handles it, but assert it:
add a `weighted_plus_bernoulli_bootstrap_still_matches_cpu` case, or explicitly record
it as covered by T23.

---

### T08 — E2e device oracle: weighted training, SymmetricTree

- **Specs**: GDC-07
- **Depends on**: T03 (`weighted_device_sym/`), T04, T07.
- **Parallel**: YES with T09/T15/T22.

**Goal / observable completion**: a real `cb_train::train(&GpuBackend::default(), …)`
fit with non-uniform weights commits to the device and its predictions match upstream
`predictions.npy` at ≤1e-5.

**Files / symbols**
- Create: `crates/cb-train/tests/device_weighted_fit_test.rs` (all §2.2 conventions).
- Read: `crates/cb-oracle/fixtures/weighted_device_sym/{X.npy,y.npy,weights.npy,model.json,predictions.npy}`.
- Borders come from `load_model_json(...).float_feature_borders()` (precedent
  `ordered_boost_e2e_oracle_test.rs:124-127`).

**Red**
- Function: `device_weighted_symmetric_fit_matches_upstream`.
- Assertions, in order:
  1. `dev.oblivious_trees.len() == params.iterations` and
     `dev.non_symmetric_trees.is_empty() && dev.region_trees.is_empty()` (the right arm
     fired — NOT a CPU fallback);
  2. `cb_model::predict_raw(&CbModel::from_trained(&dev, borders), &columns)` vs
     `predictions.npy`, `max|Δ| ≤ 1e-5`.
- Expected failure before T04+T07: either the fit falls back to CPU (assertion 1 fails)
  or the leaf values are unweighted (assertion 2 fails at ~1e-1).
- Run: `cargo check -p cb-train --no-default-features --features rocm --test device_weighted_fit_test`
  then `cargo test -p cb-train --no-default-features --features rocm --test device_weighted_fit_test`.

**Green**
- No production change — T04/T07 already landed. If it fails, the fix belongs in
  T04/T07, never here.

**Refactor**
- Factor `fixture(rel)` / `load_columns` helpers to mirror the precedent files.

**Verify**
- `cargo test -p cb-train --no-default-features --features rocm --test device_weighted_fit_test`
- `cargo check -p cb-train --no-default-features --features rocm --test device_weighted_fit_test`
- Confirm: SKIP path prints and returns on cpu/wgpu.

**Completion evidence**: ≤1e-5 max divergence printed and asserted; arm-routing
assertion green. **Acceptance scenario 2 (SymmetricTree half).**

---

### T09 — E2e device oracle: weighted training, Depthwise / Lossguide / Region

- **Specs**: GDC-08
- **Depends on**: T03 (`weighted_device_nonsym/`), T05, T06, T07.
- **Parallel**: YES with T08/T15/T22.

**Goal / observable completion**: each of the three non-symmetric policies commits to
the device under non-uniform weights and matches a CPU reference at ε=1e-4; Depthwise
additionally matches upstream `predictions.npy` at ≤1e-5.

**Files / symbols**
- Create: `crates/cb-train/tests/device_weighted_nonsym_fit_test.rs`.
- Structure: `device::run(grow_policy, label)` mirroring
  `device_nonsym_fit_test.rs:124-178` exactly (device fit vs local `CpuRefRuntime`
  fit, tree-kind routing assertions, ε=1e-4 prediction comparison), plus the upstream
  ≤1e-5 comparison for the Depthwise case only.
- Three `#[test]` wrappers with the `#[cfg(...)] / #[cfg(not(...))] SKIP` pattern.

**Red**
- Functions: `device_weighted_depthwise_matches_upstream`,
  `device_weighted_lossguide_matches_cpu`, `device_weighted_region_matches_cpu`.
- Expected failure before T05/T06/T07: CPU fallback (routing assertion) or unweighted
  leaf values (ε assertion).
- Run: `cargo check -p cb-train --no-default-features --features rocm --test device_weighted_nonsym_fit_test`
  then the `cargo test` form.

**Green**
- No production change (T05/T06/T07 own the behaviour).
- If Lossguide or Region diverges from the CPU reference beyond ε=1e-4, the fix is in
  T05/T06 — and only then generate the extra upstream scenario deferred in T03.

**Refactor**
- Share the `weight` construction and the fixture loader with T08 via a small
  `mod common` inside each test file (integration tests cannot share a module across
  binaries without a `tests/common/mod.rs` — use that if duplication exceeds ~40 lines).

**Verify**
- `cargo test -p cb-train --no-default-features --features rocm --test device_weighted_nonsym_fit_test`
- `cargo test -p cb-train --no-default-features --features rocm --test device_nonsym_fit_test`
- `cargo test -p cb-train --no-default-features --features rocm --test device_region_fit_test`

**Completion evidence**: 3 policies green. **Acceptance scenario 2 (non-symmetric half).**

---

### T10 — `DeviceCtrAveraging` type + `DeviceCtrConfig.averaging` field

- **Specs**: GDC-09 (type half)
- **Depends on**: none — **WAVE 0**.
- **Parallel**: YES; shares `cb-compute/src/runtime.rs` with T02 (different items).

**Goal / observable completion**: `DeviceCtrConfig` carries an optional second
(averaging-permutation) column set, as a plain host type; every existing construction
site still compiles (`..Default::default()` / explicit `None`).

**Files / symbols (verified, V-10)**
- Modify: `crates/cb-compute/src/runtime.rs:1067-1077` — add
  `pub averaging: Option<DeviceCtrAveraging>` to `DeviceCtrConfig`.
- Create (same file):
  ```rust
  #[derive(Debug, Clone, Default, PartialEq)]
  pub struct DeviceCtrAveraging {
      pub permutation: Vec<u32>,          // cat_averaging_permutation
      pub target_class: Vec<u32>,         // SAME shape as DeviceCtrConfig.target_class (Vec<u32>, NOT u8)
      pub columns: Vec<DeviceCtrColumn>,  // materialized under the averaging permutation
  }
  ```
  (`target_class` is included because `launch_ordered_ctr_resident` requires it per
  permutation — verified at `session.rs:172-180`. SPEC §4 omitted it.)
- No `cubecl` import; `DeviceCtrConfig` keeps `Debug + Clone + Default + PartialEq`.

**Red**
- Test: `crates/cb-compute/src/runtime_ctr_averaging_test.rs` (new sibling file),
  function `device_ctr_config_carries_optional_averaging`.
- Assert: `DeviceCtrConfig::default().averaging.is_none()`; a constructed
  `Some(DeviceCtrAveraging { … })` round-trips through `Clone`/`PartialEq`.
- Expected failure: field does not exist (compile error).
- Run: `cargo test -p cb-compute -- runtime_ctr_averaging`

**Green**
- Add the field and struct; update any exhaustive struct literal (grep
  `DeviceCtrConfig {` across `crates/`) to add `averaging: None`.

**Refactor**
- Doc-comment the invariant SPEC GDC-09 states: for a covered CTR fit `averaging` is
  always `Some`, never partially populated; `None` means "not a covered CTR regime".

**Verify**
- `cargo test -p cb-compute`
- `cargo check -p cb-backend --no-default-features --features rocm`
- `cargo check -p cb-train --no-default-features --features rocm --test device_nonsym_fit_test`

**Completion evidence**: field present, default `None`, workspace builds.

---

### T11 — Materialize the averaging-permutation CTR columns in `begin()`

- **Specs**: GDC-09 (behaviour half)
- **Depends on**: T10. Coordinate file-order with T16 (both edit `begin`).
- **Parallel**: NO vs T16 (same function, different regions).

**Goal / observable completion**: for a covered CTR fit with `averaging: Some(...)`,
`begin()` produces a SECOND resident binarized cindex column set, bit-exact vs the
CPU `averaging_ctr_features` reference, held separately from the structure columns and
never read by the split scorer.

**Files / symbols (verified, V-10)**
- Modify: `crates/cb-backend/src/gpu_runtime/session.rs:157-188` — change
  `build_ctr_cindex_columns(client, ctr: &DeviceCtrConfig, n)` to take
  `(client, permutation: &[u32], target_class: &[u32], columns: &[DeviceCtrColumn], n)`
  so it can be called twice. Update the existing call site.
- Modify: `crates/cb-backend/src/gpu_runtime/session.rs:130-144` `ctr_covered` — extend
  the shape invariants to the averaging set: `averaging.is_some()`,
  `averaging.permutation.len() == n`, `averaging.target_class.len() == n`,
  `averaging.columns.len() == columns.len()`, same `borders.len() + 1 == n_bins` rule.
- Modify: the CTR arm of `begin` (the site that currently calls
  `build_ctr_cindex_columns` and augments the resident cindex) — call it a second time
  with the averaging inputs and store the result.
- **Storage shape (SPEC GDC-09's Unresolved item — DECIDED)**: store a
  `ctr_averaging_bins: Option<Vec<Vec<u32>>>` (host `Vec`, one column per CTR column,
  object order) on `GpuTrainSession`, **separate** from the resident cindex. Rationale:
  (a) it structurally guarantees the split scorer cannot read it (the scorer only sees
  the packed resident cindex — verified `session.rs:613-618` /
  `n_features_effective()`), which is GDC-09's defining invariant; (b) T12's leaf
  gather is a per-object re-derivation that already needs host-side access to the
  tree's split list; (c) it costs `n_ctr_columns · n · 4` bytes, negligible at fixture
  scale and bounded by the same single-fit lifetime as the structure columns.
  A resident-handle variant is a later perf option, not a correctness requirement.

**Red**
- Extend `crates/cb-backend/src/kernels/ctr_device_test.rs` (existing verified tests:
  `ordered_ts_matches_cpu_reference`, `first_doc_in_segment_reads_prior`,
  `one_hot_ctr_matches_cpu_reference`, `tensor_combination_ctr_matches_cpu_reference`,
  `ctr_binarized_cindex_column_bit_exact`).
- New: `ctr_averaging_permutation_column_bit_exact` — same structure as
  `ctr_binarized_cindex_column_bit_exact` but driven by a DIFFERENT permutation, with a
  CPU-transcribed `averaging_ctr_features` reference
  (`crates/cb-train/src/boosting.rs:3906-3920` is the semantic source; transcribe it,
  do not depend on `cb-train` from `cb-backend`).
- Additionally a session-level test in
  `crates/cb-backend/src/gpu_runtime/session_depth_gt1_test.rs` (precedent:
  `session_ctr_gate_covers_single_permutation`, `session_ctr_augments_resident_cindex`,
  `:478-620`): `session_ctr_materializes_averaging_columns_separately` asserting
  (i) `n_features_effective()` is UNCHANGED by the averaging columns (they do not join
  the scorer's cindex), and (ii) the averaging column values differ from the structure
  column values for at least one object (proving the two permutations really diverge —
  a degenerate fixture where they agree cannot discriminate T12's bug).
- Expected failure: no averaging materialization exists.
- Run: `cargo test -p cb-backend --no-default-features --features rocm -- ctr_`

**Green**
- Refactor `build_ctr_cindex_columns`'s parameters, call it twice, store the second
  result in the new session field. Reuse `launch_ordered_ctr_resident` +
  `binarize_ctr_column_resident` verbatim (Do Not Hand-Roll).
- `averaging: None` ⇒ byte-unchanged (regression guard).

**Refactor**
- Fold the two call sites into one private
  `materialize_ctr_columns(client, perm, tc, cols, n)` helper if the diff duplicates
  more than the argument list.

**Verify**
- `cargo test -p cb-backend --no-default-features --features rocm`
- `cargo check -p cb-backend --no-default-features --features rocm`
- Confirm: every non-CTR device path is byte-unchanged.

**Completion evidence**: both new self-oracles green; `n_features_effective()`
unchanged; the divergence assertion proves the fixture discriminates.

**Risks**: extending `ctr_covered` could newly DECLINE a currently-covered CTR session
(the existing `session_ctr_gate_covers_single_permutation` test constructs a config
with no `averaging`). **Guardrail**: that test must be updated deliberately (either give
it an `averaging`, or gate the new invariants on `averaging.is_some()`), and the change
recorded — do not weaken the assertion to make it pass.

---

### T12 — Device leaf-value gather from the averaging-permutation CTR bins

- **Specs**: GDC-10 — **the highest-uncertainty task in this plan** (SPEC §9 says so;
  verification did not find any existing precedent for gathering leaf sums from a second
  bin set).
- **Depends on**: T11 (averaging columns resident/stored), T04 (same leaf-reduce region
  of `mod.rs`).
- **Parallel**: NO vs T04/T11.

**Goal / observable completion**: for a tree containing ≥1 CTR-based split, leaf values
are computed over a leaf assignment re-derived from the AVERAGING-permutation CTR bins
(matching CPU `assign_leaf_over_ctr_columns`, `boosting.rs:5816-5817` / `:1935`), while
split SEARCH still uses the structure-permutation bins. For a tree with no CTR split the
behaviour is byte-unchanged.

**Files / symbols (verified)**
- CPU parity target to transcribe: `crates/cb-train/src/boosting.rs:1935`
  `assign_leaf_over_ctr_columns(&matrix, &averaging_ctr_features, &grown, n)`, called at
  `:5817` and `:6120`.
- Modify: `crates/cb-backend/src/gpu_runtime/mod.rs:4227-4264` — the leaf-stat reduce
  and the host `calc_average` loop.
- Modify: `crates/cb-backend/src/gpu_runtime/session.rs` `grow_one` oblivious arm — after
  the grow returns `tree.splits` + `tree.leaf_of`, when `ctr_averaging_bins.is_some()`
  and at least one chosen split's `feature_id >= n_features_base` (i.e. it is a CTR
  column, since CTR columns are appended after the float|one-hot axis — verified
  `session.rs:613-618`), re-derive `leaf_of_avg` on the host from the averaging bins and
  recompute the leaf values.

**Design (decided here; SPEC left it open)**
1. Keep the device grow exactly as today: it returns `splits`, `leaf_of` (structure
   bins) and `part_stats`.
2. On the host, when the tree has ≥1 CTR split: rebuild the per-object leaf index by
   replaying `tree.splits` in forward bit order, substituting — for the CTR-column
   splits only — the object's **averaging** bin instead of its structure bin. This is a
   direct transcription of `assign_leaf_over_ctr_columns` and needs no new kernel.
3. Recompute `leaf_values[leaf] = calc_average(Σ(w·der1) over leaf_of_avg,
   Σw over leaf_of_avg, scaled_l2)` using `cb_core::sum_f64` over the host der/weight
   (obtainable via `host_der1(self.der_kernel, approx, target)` — the same helper the
   nonsym/region/exact arms already use, verified `session.rs:1682, 1711, 1915`).
4. Emit the recomputed `leaf_values` on the returned `DeviceGrownTree`; `leaf_of` stays
   the STRUCTURE assignment (the model's apply path re-derives membership from splits).
   **Confirm this last point against `cb_model::predict_raw` before finalising** — if the
   apply path needs the averaging assignment, escalate (see §6 Q3).

This keeps D-05 intact: the extra host work is `O(n · depth)` once per tree, the same
crossing class as the already-existing single `leaf_of` read-back.

**Red**
- New self-oracle in `crates/cb-backend/src/gpu_runtime/session_depth_gt1_test.rs`:
  `ctr_leaf_values_use_averaging_permutation_bins`.
- Setup: a session with a CTR config whose structure and averaging permutations
  **provably diverge** (reuse T11's divergence assertion as a precondition), a tree
  forced to contain a CTR split, and a hand-computed CPU reference doing exactly the
  `assign_leaf_over_ctr_columns` re-assignment.
- Assertion: device leaf values match the CPU reference at ε=1e-4 **and** differ from
  the structure-only computation by more than 1e-4 (the second half is what makes this
  test discriminating — research.md pitfall #2).
- Expected failure: leaf values equal the structure-only value.
- Run: `cargo test -p cb-backend --no-default-features --features rocm -- ctr_leaf_values`

**Green**
- Implement steps 1-4. No new CubeCL kernel.

**Refactor**
- Extract `fn reassign_leaves_over_ctr(splits, structure_bins, averaging_bins, n) -> Vec<u32>`
  as a private helper with its own doc reference to the CPU source line.
- Ensure no `unwrap`/indexing (D-13).

**Verify**
- `cargo test -p cb-backend --no-default-features --features rocm`
- Confirm: a float-only / one-hot-only tree takes the untouched path (assert the
  existing `session_residency` oracles are byte-unchanged).

**Completion evidence**: the discriminating self-oracle green; non-CTR paths unchanged.

**Risks**: highest in the plan. If the recomputation cannot reproduce the CPU reference,
the failure belongs here, not in T15. **Escalation trigger**: if step 4's `leaf_of`
contract turns out to be wrong (see §6 Q3), stop and report rather than changing
`DeviceGrownTree`'s meaning.

---

### T13 — NEW mixed float+cat CTR e2e fixture (`ctr_device_mixed/`)

- **Specs**: GDC-06 (CTR scenario) / prerequisite for GDC-12
- **Depends on**: §6 Q2 sign-off (this is a scope addition SPEC did not anticipate —
  see **B-5**).
- **Parallel**: YES — touches only `crates/cb-oracle/`.

**Goal / observable completion**: a frozen upstream fixture exists whose configuration
can actually reach the device CTR arm.

**Why (verified, V-9)**: `plain_ctr/` has no trained model at all; every other CTR e2e
fixture is cat-only, and a cat-only pool yields `device_n_features == 0` /
`device_n_bins == 0` (`boosting.rs:4208-4209, 4292`) so it can never commit to the
device. SPEC GDC-12's "reuse rather than regenerate" is not achievable.

**Files / symbols**
- Create: `crates/cb-oracle/fixtures/ctr_device_mixed/{gen_fixtures.py,config.json,X.npy,X_cat.npy,y.npy,model.json,predictions.npy}`

**Pinned recipe (every value load-bearing; two constraints are hard gate requirements)**
```
catboost==1.2.10, RandomState(0), thread_count=1
n_rows = 64
n_float_features = 2,  border_count = 15     # -> 16 float bins  (HARD: must equal ctr bins)
n_cat_features   = 1,  cardinality 6
one_hot_max_size = 1                         # routes the cat column to CTR, not one-hot
simple_ctr = ["Borders:Prior=0.5"], combinations_ctr = [], max_ctr_complexity = 1
ctr_border_count = 15                        # -> 16 CTR bins   (HARD: borders.len()+1 == n_bins)
counter_calc_method = "SkipTest"
loss_function = "Logloss", iterations = 5, depth = 2, learning_rate = 0.1, l2_leaf_reg = 3.0
boosting_type = "Plain", permutation_count = 1, fold_len_multiplier = 2.0
bootstrap_type = "No", random_strength = 0, boost_from_average = False
leaf_estimation_method = "Gradient", leaf_estimation_iterations = 1
```
- `[V]` the `borders.len() + 1 == n_bins` requirement is `ctr_covered`,
  `session.rs:140-143`; `[V]` `ctr_border_count_default() == 15`,
  `boosting.rs:573-575`.
- **Anti-false-pass guards in the generator** (all three mandatory):
  1. assert the trained model actually contains ≥1 CTR split (else the fixture is
     vacuous — the existing CTR generators already do this);
  2. assert ≥1 float split too (else the float axis is decorative and the device
     `n_features` argument would be untested);
  3. assert the structure and averaging permutations **differ** at this `n`/seed
     (SPEC GDC-12's own Unresolved item). Emit
     `permutation_structure.npy` / `permutation_averaging.npy` alongside so the Rust
     side can re-assert it. If they agree, change the seed until they do not, and record
     which seed was used and why.

**Red**
- Test: `crates/cb-oracle/tests/ctr_device_mixed_fixture_smoke_test.rs`,
  function `ctr_device_mixed_fixture_is_device_shaped`.
- Assert: 2 float columns × 15 borders from `model.json`, 1 cat column,
  `predictions.npy` length 64, and the two permutation arrays differ in ≥1 position.
- Expected failure: directory absent.
- Run: `cargo test -p cb-oracle --test ctr_device_mixed_fixture_smoke_test`

**Green**
- Generate offline against installed `catboost==1.2.10`; commit; mark FROZEN.

**Refactor**
- Mirror `ctr_btmv_simple/gen_fixtures.py`'s docstring structure (pinned recipe,
  reproducibility caveat, why-this-fixture-exists).

**Verify**
- `cargo test -p cb-oracle --test ctr_device_mixed_fixture_smoke_test`

**Completion evidence**: fixture committed; permutation-divergence assertion green
(this is what makes T15 able to discriminate a structure-only bug).

---

### T14 — Populate `DeviceTrainConfig.ctr` (both permutations) and relax the CTR clauses

- **Specs**: GDC-11
- **Depends on**: T01, T07 (gate chain order), T11, T12.
- **Parallel**: NO — second of the serial gate chain.

**Goal / observable completion**: a single-permutation, CPU-legal-type CTR fit with at
least one float feature commits to the device (`device_active == true`); a
`permutation_count > 1` CTR fit still does not.

**Files / symbols (verified)**
- Modify: `crates/cb-train/src/boosting.rs:4085-4086` — replace
  `materialized_ctr_features.is_empty() && structure_fold_columns.iter().all(Vec::is_empty)`
  with
  `(materialized_ctr_features.is_empty() || (learning_folds_for_cycle == 1 && ctr_types_are_device_covered(&materialized_ctr_features)))`.
- Create (same file, private): `fn ctr_types_are_device_covered(cols: &[CtrFeatureColumn]) -> bool`
  restricting to Borders / Buckets / BinarizedTargetMeanValue / Counter-under-SkipTest
  (mirror `CPU_LEGAL_CTR_TYPES` in `crates/catboost-rs-py/src/params.rs`).
- Modify: `crates/cb-train/src/boosting.rs:4226-4291` — populate
  `ctr: Some(DeviceCtrConfig { permutation: cat_learn_permutation.into(), target_class,
  columns: <from materialized_ctr_features>, averaging: Some(DeviceCtrAveraging {
  permutation: cat_averaging_permutation.into(), target_class, columns: <from
  averaging_ctr_features> }) })` when CTR is device-covered, `None` otherwise.
  Source bindings all exist already at `boosting.rs:3662-3920` (verified:
  `structure_fold_columns` `:3837-3890`, `materialized_ctr_features` `:3894-3897`,
  `averaging_ctr_features` `:3906-3920`, the two permutations `:3721-3750`).
- Note the permutation type conversion: CPU permutations are `Vec<i32>`;
  `DeviceCtrConfig.permutation` is `Vec<u32>` (verified `runtime.rs:1072`). Convert with
  a checked cast that rejects negatives (no `as` truncation, no `unwrap`).

**Red**
- Test: `crates/cb-train/tests/device_ctr_gate_test.rs`.
- `single_permutation_ctr_commits_to_device`: the T13 fixture's config, a `CountingGpu`
  runtime, assert `iterations` device grow calls.
- `multi_permutation_ctr_declines_to_device`: same but `permutation_count = 4`, assert
  **zero** device grow calls (this is the GDC-01 regression proof, and closes
  **acceptance scenario 3**).
- Expected failures: zero grow calls in the first (clause still excludes CTR); the
  second passes trivially before the change and must keep passing after.
- Run: `cargo check -p cb-train --no-default-features --features rocm --test device_ctr_gate_test`
  then the `cargo test` form.

**Green**
- The clause relaxation + config population + the type helper. Nothing in `cb-backend`.

**Refactor**
- Hoist the `DeviceCtrConfig` construction into a private
  `fn build_device_ctr_config(...) -> Option<DeviceCtrConfig>` so `device_config`'s
  literal stays readable.

**Verify**
- `cargo test -p cb-train --no-default-features --features rocm --test device_ctr_gate_test`
- `cargo test -p cb-train --no-default-features --features rocm --test device_weight_gate_test`
  (T07 must not have regressed)
- `cargo test --workspace` — the CPU CTR oracles
  (`ctr_btmv_simple_oracle_test`, `ctr_structure_fold_shuffle_test`,
  `ctr_btmv_bake_upstream_table_test`, …) must stay green: under default features
  `GpuBackend` is not compiled, so they still take the CPU path.
- Confirm: one-hot × CTR stays excluded (`is_covered_regime`'s
  `one_hot_flags.iter().all(|&f| !f) || ctr.is_none()`, `runtime.rs:1208` — untouched).

**Completion evidence**: both gate tests green; CPU CTR oracles unchanged.
**Acceptance scenario 3.**

**Risks**: `permutation_count > 1` CTR silently riding fold-0 columns — prevented by
the `learning_folds_for_cycle == 1` guard, itself only meaningful because T01 threaded
the real value. **Guardrail**: `multi_permutation_ctr_declines_to_device` must be
verified to fail if T01 is reverted (state this in the completion note).

---

### T15 — E2e device oracle: CTR training, structure AND leaf values

- **Specs**: GDC-12
- **Depends on**: T13, T14.
- **Parallel**: YES with T08/T09/T22.

**Goal / observable completion**: a real `train_cat(&GpuBackend::default(), …)` CTR fit
commits to the device and its predictions match upstream at ≤1e-5 — the bar a
structure-only implementation fails.

**Files / symbols**
- Create: `crates/cb-train/tests/device_ctr_fit_test.rs` (all §2.2 conventions).
- Entry point: `cb_train::train_cat` (`boosting.rs:2614`) — returns
  `(Model, BakedCtrData)`; attach the baked data via `cb_model::CtrData::from_baked` +
  `Model::with_ctr_data` before predicting (precedent: the existing CPU CTR oracle
  tests, e.g. `crates/cb-train/tests/ctr_btmv_simple_oracle_test.rs`).
- Fixture: `crates/cb-oracle/fixtures/ctr_device_mixed/`.

**Red**
- Function: `device_ctr_fit_matches_upstream_predictions`.
- Assertions, in order:
  1. `dev.oblivious_trees.len() == params.iterations` (right arm fired, no CPU fallback);
  2. the model contains ≥1 CTR split (else the test is vacuous);
  3. `predict_raw` vs `predictions.npy`, `max|Δ| ≤ 1e-5`.
- Additionally `device_ctr_leaf_values_differ_from_structure_only`: assert the device
  leaf values differ from a structure-only recomputation by > 1e-4 (the pitfall-#2
  discriminator, complementing T12's unit-level version at the e2e level).
- Expected failure before T12/T14: CPU fallback, or ≥1e-1 prediction divergence.
- Run: `cargo check -p cb-train --no-default-features --features rocm --test device_ctr_fit_test`
  then the `cargo test` form.

**Green**
- No production change. Failures belong to T11/T12/T14.

**Refactor**
- Share the fixture loader with T08/T09 via `tests/common/mod.rs` if warranted.

**Verify**
- `cargo test -p cb-train --no-default-features --features rocm --test device_ctr_fit_test`
- `cargo test --workspace` (CPU CTR oracles unchanged)

**Completion evidence**: ≤1e-5 printed and asserted; the discriminator assertion green.
**Acceptance scenario 4.**

---

### T16 — Consume `map_ordered_coverage`'s result in `begin()`

- **Specs**: GDC-14
- **Depends on**: T02 (so the seam already carries the parameter).
- **Parallel**: NO vs T11 (same function, ~250 lines apart) — sequence T16 before T11.

**Goal / observable completion**: `begin()` no longer unconditionally declines every
Ordered fit; a covered `OrderedState` is stored on the session and construction
proceeds through the remaining (unchanged) gate clauses.

**Files / symbols (verified)**
- Modify: `crates/cb-backend/src/gpu_runtime/session.rs:988-991` — replace
  ```rust
  if !boosting_type_is_plain {
      let _ordered = map_ordered_coverage(loss, config, depth, fold_count);
      return Ok(None);
  }
  ```
  with a binding that declines only on `None`:
  ```rust
  let ordered_state = if boosting_type_is_plain { None } else {
      match map_ordered_coverage(loss, config, depth, fold_count) {
          Some(s) => Some(s),
          None => return Ok(None),
      }
  };
  ```
  and carry `ordered_state` into the constructed `GpuTrainSession`
  (`ordered` field, `session.rs:827-835`, currently always `None`).
- Remove `#[allow(dead_code)]` from `OrderedState` (`session.rs:526`) once it is live.
- `map_ordered_coverage` itself (`session.rs:543-566`) is UNCHANGED — it already
  requires `depth >= 1`, `fold_count == 1`, `SymmetricTree`, and
  `config.ctr.is_none()` (the last is V-11's backend-side Ordered×CTR exclusion).
- Every remaining `begin` clause (depth, loss, score fn, bootstrap, n/n_features/n_bins)
  still applies identically to the Ordered path — do not add a cross-family exemption.

**Red**
- This intentionally FLIPS the existing pinned test, which is why GDC-18/T17 exists as
  a separate spec. In T16 write the NEW positive test first:
  `crates/cb-backend/src/gpu_runtime/ordered_test.rs`, function
  `begin_admits_covered_ordered_config`.
- Setup: reuse `begin_declines_ordered_to_cpu`'s (`ordered_test.rs:250-313`) exact
  covered-RMSE inputs, `boosting_type_is_plain = false`.
- Assertion: `GpuTrainSession::begin(...)` returns `Ok(Some(_))`.
- Expected failure: `Ok(None)` from the unconditional early return.
- Run: `cargo test -p cb-backend --no-default-features --features rocm -- begin_admits_covered_ordered_config`

**Green**
- The binding change above. Do NOT wire the grow (T18/T19) and do NOT relax
  `boosting.rs` (T21) — after this task an Ordered fit still cannot reach `begin` from
  `cb-train`, so the change is observable only from the backend self-oracle. That is
  intentional and is what makes T16 independently testable.

**Refactor**
- Rewrite the long comment block at `session.rs:974-987` to describe the new behaviour
  (covered ⇒ proceed, uncovered ⇒ decline) instead of the removed forward-dependency
  rationale.

**Verify**
- `cargo test -p cb-backend --no-default-features --features rocm`
  — expect `begin_declines_ordered_to_cpu` to FAIL here; that is T17's job to fix and
  must not be papered over by weakening it.
- `cargo check -p cb-backend --no-default-features --features rocm`

**Completion evidence**: the new positive test green; the known-failing
`begin_declines_ordered_to_cpu` explicitly recorded as T17's input (do not merge T16
without T17).

---

### T17 — Update `begin_declines_ordered_to_cpu` to the new, correct behaviour

- **Specs**: GDC-18
- **Depends on**: T16.
- **Parallel**: NO vs T16.

**Goal / observable completion**: both directions stay tested — a covered ordered
config admits, an UNCOVERED one still declines. The negative case must not be lost
while fixing the positive one.

**Files / symbols (verified)**
- Modify: `crates/cb-backend/src/gpu_runtime/ordered_test.rs:250-313`.

**Red**
- Rename/repurpose into a pair:
  - `begin_admits_covered_ordered_config` (T16's test — moves here as the canonical
    positive);
  - `begin_declines_uncovered_ordered_config` — same call with an input
    `map_ordered_coverage` rejects. Use a coverage-breaking input that is NOT also
    rejected by an earlier clause, e.g. `config.max_leaves = Some(k)` or
    `config.exact_leaf = true` (both are checked in `map_ordered_coverage`'s
    `family_default`, `session.rs:557-561`) — assert `Ok(None)`.
- Expected failure: the file currently has one test asserting the opposite of the
  positive case.
- Run: `cargo test -p cb-backend --no-default-features --features rocm -- ordered_`

**Green**
- Rewrite the assertions. **Do not weaken** — the negative case must use a genuinely
  uncovered config, not a trivially-rejected one (e.g. `depth == 0` also declines via
  a different clause and would not exercise `map_ordered_coverage`).

**Refactor**
- Add a comment recording that this flip is intentional (GDC-18) and citing T16.

**Verify**
- `cargo test -p cb-backend --no-default-features --features rocm`
  — the full cb-backend self-oracle set green again, including the other four ordered
  tests (`device_trajectory_matches_frozen_cpu_reference`, `body_rows_keep_delta_zero`,
  `resident_trajectory_persists_across_iterations`,
  `single_tail_row_equals_gradient_leaf_delta`).

**Completion evidence**: both directions green. **Acceptance scenario 9.**

---

### T18 — Ordered per-segment split scoring on the device grow path — **BLOCKED**

- **Specs**: GDC-15 (structure half)
- **Depends on**: T02, T16, **and §6 Q1 sign-off**.
- **Parallel**: NO vs T19/T21.
- **Status**: **BLOCKED — decision required before execution.** See **B-3 / V-7** and
  §6 Q1.

**Why blocked**: verification shows the CPU Ordered path differs from Plain ONLY in
`greedy_tensor_search_oblivious_ordered` (`boosting.rs:5583-5591` →
`tree.rs:2498-2548` → `select_level_ordered` `:2427` → `score_candidate_ordered`
`:2383-2418`), and that `ordered_approx_delta_simple` — the primitive the shipped device
driver `accumulate_ordered_trajectory` reproduces — **is never called in the training
loop** (`grep` across `boosting.rs`, `tree.rs`, `lib.rs`). Implementing GDC-15 exactly
as written (trajectory apply keyed by the real `leaf_of`) would therefore NOT reproduce
the CPU/upstream ordered fit, and T22's ≤1e-5 bar would fail for a reason no task in
this plan owns.

**Goal / observable completion (assuming Q1 resolves toward "match the CPU path")**:
the device Ordered arm chooses the SAME per-level splits as
`greedy_tensor_search_oblivious_ordered` for a fixed input, at integer equality on the
chosen `(feature, border)` per level.

**Files / symbols (verified)**
- CPU parity source to transcribe: `crates/cb-train/src/tree.rs:2383-2418`
  (`score_candidate_ordered`: per-segment `scale_l2_reg(l2, body_sum_weight,
  body_finish)`, per-segment `l2_split_score`, `cb_core::sum_f64` across segments) and
  `:2427-2534` (`select_level_ordered` candidate order + strict-`>` first-wins argmax
  via `select_best_candidate`).
- Segment inputs: `crates/cb-train/src/fold.rs:105` `body_tail_boundaries(n, mult)`,
  `:137` `body_tail_segments`, `:368` `body_sum_weights` — all pure `(n, multiplier)`
  functions computed **cb-train-side** and passed through `OrderedTreeArgs` (T02) so
  `cb-backend` never depends on `cb-train` (T-10-04).
- Likely new: `crates/cb-backend/src/kernels/ordered_grow.rs` (host-driven, mirroring
  `region_device.rs` / `nonsym_grow.rs`: host controls the level loop, the device scores
  candidates). **Do not** author a new CubeCL kernel before checking whether
  `device_best_split_for_node` / the existing partition histogram can be driven
  per-segment; consult
  `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` first if any
  kernel code is needed (AGENTS.md mandate).

**Red**
- New self-oracle `crates/cb-backend/src/kernels/ordered_grow_test.rs`,
  `ordered_device_split_choice_matches_cpu_reference`: a frozen small fixture
  (n=30, multiplier=2.0 ⇒ segments `[(1,2),(2,4),(4,8),(8,16),(16,30)]` per the
  documented example at `fold.rs:135`), a transcribed CPU reference, assert the chosen
  `(feature, border)` per level is **integer-equal** and the summed score matches at
  ε=1e-4.
- Expected failure: no ordered device grow exists.
- Run: `cargo test -p cb-backend --no-default-features --features rocm -- ordered_grow`

**Green**
- Implement the per-segment scored level search, reusing `cb_compute::l2_split_score`,
  `cb_compute::scale_l2_reg`, `cb_core::sum_f64`. Note the ordered path uses **L2 only**
  (verified: the ordered fixture pins `score_function: "L2"`, and
  `score_candidate_ordered` hard-codes `l2_split_score`) — decline any other score
  function to CPU rather than approximating it.

**Refactor**
- Keep the segment loop and the candidate loop separable so T19 can reuse the level
  search unchanged.

**Verify**
- `cargo test -p cb-backend --no-default-features --features rocm -- ordered_`
- `cargo test -p cb-backend --no-default-features --features rocm`

**Completion evidence**: integer-equal split choices vs the CPU reference on the frozen
fixture.

---

### T19 — Wire the ordered arm into `grow_one` and consume `OrderedTreeArgs`

- **Specs**: GDC-15 (wiring half)
- **Depends on**: T18. Serialise after T04/T05/T06 (same function).
- **Parallel**: NO.

**Goal / observable completion**: `GpuBackend::grow_tree_on_device(..., Some(&OrderedTreeArgs))`
grows an ordered tree through the T18 search over the session's resident state and
returns a `DeviceGrownTree` whose splits and leaf values match the CPU ordered fit at
ε=1e-4 across multiple trees.

**Files / symbols (verified)**
- Modify: `crates/cb-backend/src/gpu_backend.rs:310-335` — forward `ordered` into
  `session.grow_one(approx, target, sample, ordered)`.
- Modify: `crates/cb-backend/src/gpu_runtime/session.rs:1624-1629` `grow_one` signature
  + a new ordered arm placed BEFORE the region/nonsym arms (they are mutually
  exclusive — `map_ordered_coverage` requires `SymmetricTree`).
- Modify: `crates/cb-train/src/boosting.rs:4598` — build `OrderedTreeArgs` from
  `ordered_learning_perm` (`boosting.rs:3562-3585`, a `Vec<i32>`) plus
  `crate::fold::body_tail_segments(n, params.fold_len_multiplier)`; pass `None` on the
  Plain path (byte-unchanged, D-04).
  Per **V-6**, `body_finish`/`tail_finish` are the segment scalars, and the whole
  `OrderedTreeArgs` is per-fit constant — construct it ONCE outside the iteration loop
  and borrow it per tree.
- Reuse `launch_apply_leaf_delta_into` for the approx advance (Do Not Hand-Roll), the
  same call the oblivious resident grow already makes at `mod.rs:4268`.

**Red**
- Extend `crates/cb-backend/src/gpu_runtime/ordered_test.rs` with
  `ordered_grow_over_real_leaf_of_matches_cpu_multi_tree`: grow 3 trees through
  `grow_one` with a real (non-identity) `leaf_of` produced by T18's search, compare the
  per-object approx trajectory and per-tree leaf values against the CPU reference at
  ε=1e-4. (The existing `device_trajectory_matches_frozen_cpu_reference` stays as the
  isolated-primitive test — do not modify it.)
- Expected failure: `grow_one` has no ordered arm.
- Run: `cargo test -p cb-backend --no-default-features --features rocm -- ordered_grow_over_real_leaf_of`

**Green**
- Implement the arm. Preserve D-05: at most one `leaf_of` + `part_stats` read-back per
  tree, never per-object per-level.

**Refactor**
- Ensure the ordered arm shares the resident approx / der1 advance discipline with the
  oblivious arm rather than duplicating it.

**Verify**
- `cargo test -p cb-backend --no-default-features --features rocm`
- `cargo check -p cb-train --no-default-features --features rocm --test device_nonsym_fit_test`
- `cargo test --workspace` — the Plain path must be byte-unchanged (`ordered == None`).

**Completion evidence**: multi-tree ordered self-oracle green at ε=1e-4; every Plain
oracle unchanged.

---

### T20 — NEW ordered device fixture (`ordered_boost_device/`, `boost_from_average=False`)

- **Specs**: GDC-06 (ordered scenario) / prerequisite for GDC-17
- **Depends on**: §6 Q1 (scope) — but the fixture itself can be generated as soon as
  Q1 confirms the phase proceeds.
- **Parallel**: YES — touches only `crates/cb-oracle/`.

**Why (verified, V-8 / B-4)**: `ordered_boost_e2e/config.json` pins
`boost_from_average: true` and `mean(y) = 1.4695…` ⇒ `bias ≈ 1.47 ≠ 0`, while
`device_host_eligible`'s `bias == 0.0` clause (`boosting.rs:4165-4170`, CR-01) is
explicitly out of scope for this phase. The existing fixture can never commit to the
device.

**Files / symbols**
- Create: `crates/cb-oracle/fixtures/ordered_boost_device/{gen_fixtures.py,config.json,X.npy,y.npy,model.json,predictions.npy}`

**Pinned recipe (identical to `ordered_boost_e2e` except the one flag)**
```
catboost==1.2.10, RandomState(0), thread_count=1
n_rows = 30, n_float_features = 2, border_count = 15
loss_function = "RMSE", boosting_type = "Ordered"
iterations = 5, depth = 2, learning_rate = 0.1, l2_leaf_reg = 3.0
permutation_count = 1, fold_len_multiplier = 2.0
bootstrap_type = "No", random_strength = 0
score_function = "L2"                      # the ordered search is L2-only (V-7)
leaf_estimation_method = "Gradient", leaf_estimation_iterations = 1
boost_from_average = False                 # <-- THE difference; makes bias == 0.0
```
- Anti-false-pass guard: assert the trained model's splits differ from a
  `boosting_type=Plain` model on the same data (otherwise the ordered path is not
  exercised).

**Red**
- Test: `crates/cb-oracle/tests/ordered_boost_device_fixture_smoke_test.rs`,
  `ordered_boost_device_fixture_has_zero_bias`: assert `config.json` has
  `boost_from_average == false` and that `model.json`'s scale/bias term is `0.0`.
- Expected failure: directory absent.
- Run: `cargo test -p cb-oracle --test ordered_boost_device_fixture_smoke_test`

**Green / Refactor / Verify**: as T03/T13 (generate offline, commit, mark FROZEN,
mirror the existing generator docstring shape).

**Completion evidence**: fixture committed with `bias == 0`; smoke test green.
**Do not modify `ordered_boost_e2e/` or its CPU oracle** — SPEC GDC-17's invariant.

---

### T21 — Relax the Ordered clause with the Ordered × CTR composition guard

- **Specs**: GDC-16
- **Depends on**: T14 (gate chain order), T18, T19.
- **Parallel**: NO — last of the serial gate chain.

**Goal / observable completion**: a covered Ordered fit with no CTR commits to the
device; an Ordered fit WITH CTR does not.

**Files / symbols (verified)**
- Modify: `crates/cb-train/src/boosting.rs:4084` — replace
  `ordered_learning_perm.is_none()` with
  `(ordered_learning_perm.is_none() || ordered_device_covered(&params.loss, params.depth, params.score_function))`.
- Add to the same expression an explicit
  `&& !(ordered_learning_perm.is_some() && !materialized_ctr_features.is_empty())`
  composition guard (SPEC GDC-16's `!(ordered_active && has_ctr)`).
- Create (private): `fn ordered_device_covered(loss, depth, score_function) -> bool`
  mirroring `map_ordered_coverage`'s host-visible preconditions (depth ≥ 1, RMSE /
  Logloss / CrossEntropy der, `EScoreFunction::L2` per V-7) — a host-side pre-check, NOT
  a replacement for the backend gate.
- **V-11 note**: `map_ordered_coverage` (`session.rs:560`) already requires
  `config.ctr.is_none()`, so the guard is defence-in-depth. Keep it anyway (fail-fast at
  the single D-10-01 decision point), and make T23's assertion target the observable
  (`device_active == false`), not the specific clause.

**Red**
- Test: `crates/cb-train/tests/device_ordered_gate_test.rs`.
- `covered_ordered_fit_commits_to_device`: T20's fixture config, `CountingGpu`, assert
  `iterations` device grow calls. Expected failure: zero (clause excludes it).
- `ordered_plus_ctr_declines_to_device`: same but with a CTR-routed cat column; assert
  zero device grow calls. Must pass before AND after.
- Run: `cargo check -p cb-train --no-default-features --features rocm --test device_ordered_gate_test`
  then the `cargo test` form.

**Green**
- The clause relaxation + guard + helper.

**Refactor**
- Re-read the whole `device_host_eligible` expression (4083-4176) after T07+T14+T21 and
  confirm exactly three clauses changed and none of the untouched ones drifted (paste
  the before/after clause list into the completion note).

**Verify**
- `cargo test -p cb-train --no-default-features --features rocm --test device_ordered_gate_test`
- `cargo test -p cb-train --no-default-features --features rocm --test device_ctr_gate_test`
- `cargo test -p cb-train --no-default-features --features rocm --test device_weight_gate_test`
- `cargo test --workspace` — `ordered_boost_e2e_oracle_test` (CPU) must stay green,
  untouched.

**Completion evidence**: both directions green; the clause-diff audit recorded.
**Acceptance scenario 6 (positive half of the guard).**

---

### T22 — E2e device oracle: Ordered boosting training

- **Specs**: GDC-17
- **Depends on**: T20, T21.
- **Parallel**: YES with T08/T09/T15.

**Goal / observable completion**: a real `cb_train::train(&GpuBackend::default(), …)`
Ordered fit commits to the device and matches upstream `predictions.npy` at ≤1e-5 over
all 5 iterations.

**Files / symbols**
- Create: `crates/cb-train/tests/device_ordered_fit_test.rs` (§2.2 conventions; a
  SEPARATE file — `ordered_boost_e2e_oracle_test.rs` imports `CpuBackend` and cannot
  compile under `--no-default-features`, verified by research.md's `cargo check`
  reproduction).
- Fixture: `crates/cb-oracle/fixtures/ordered_boost_device/` (T20), **not**
  `ordered_boost_e2e/` (V-8).
- Borders from `load_model_json(...).float_feature_borders()`.

**Red**
- Function: `device_ordered_fit_matches_upstream_predictions`.
- Assertions: (1) `dev.oblivious_trees.len() == params.iterations` + empty
  nonsym/region (right arm fired); (2) `predict_raw` vs `predictions.npy`,
  `max|Δ| ≤ 1e-5` over all objects/iterations.
- Expected failure before T18/T19/T21: CPU fallback, or split-structure divergence
  producing ≫1e-5.
- Run: `cargo check -p cb-train --no-default-features --features rocm --test device_ordered_fit_test`
  then the `cargo test` form.

**Green**
- No production change; failures belong to T18/T19/T21.

**Refactor**
- Mirror the CPU oracle's assertion style so the two files are trivially diffable.

**Verify**
- `cargo test -p cb-train --no-default-features --features rocm --test device_ordered_fit_test`
- `cargo test -p cb-train --test ordered_boost_e2e_oracle_test` (the CPU sibling, still
  green and untouched)

**Completion evidence**: ≤1e-5 over all 5 iterations. **Acceptance scenario 5.**

---

### T23 — Cross-gap composition regression tests

- **Specs**: GDC-19
- **Depends on**: T07, T14, T21 (all three relaxations landed).
- **Parallel**: NO — must run last.

**Goal / observable completion**: three independently-checkable composition assertions
hold, proving no gap widened an unrelated exclusion.

**Files / symbols**
- Create: `crates/cb-train/tests/device_gate_composition_test.rs` (§2.2 conventions;
  use a `CountingGpu`-style wrapper to observe `device_active` — precedent
  `bootstrap_dev_oracle_test.rs:261-267`).

**Red — three functions, one per SPEC GDC-19 item**
1. `ordered_plus_ctr_still_declines` — `boosting_type=Ordered` + CTR-routed cat column;
   assert **zero** device grow calls. (V-11: this must hold via the host guard AND the
   backend `map_ordered_coverage` `ctr.is_none()`; assert the observable.)
   → **acceptance scenario 6**
2. `weighted_plus_ctr_admits_together` — non-uniform weights + the T13 mixed CTR
   fixture; assert `iterations` device grow calls, AND predictions match upstream at
   ≤1e-5. This is a **POSITIVE** test (research.md and
   `session_ctr_augments_resident_cindex` confirm CTR columns are just extra feature
   columns the weighted histogram fill reads normally) — do not conflate it with (1).
   → **acceptance scenario 7**
3. `depthwise_plus_bayesian_bootstrap_still_declines` — the untouched
   bootstrap × grow_policy cross-product (`boosting.rs:4118-4125`); assert **zero**
   device grow calls.
   → **acceptance scenario 8**
- Expected failures before T07/T14/T21: (2) fails (CPU fallback); (1) and (3) pass
  trivially and must keep passing.
- Run: `cargo check -p cb-train --no-default-features --features rocm --test device_gate_composition_test`
  then the `cargo test` form.

**Green**
- **No production code.** GDC-19 is verification-only: if an assertion fails, the fix
  belongs in T07 / T14 / T21, never here.

**Refactor**
- Extract the `CountingGpu` wrapper into `tests/common/mod.rs` if it is now duplicated
  across ≥3 of this phase's new test files.

**Verify**
- `cargo test -p cb-train --no-default-features --features rocm --test device_gate_composition_test`
- FULL device suite, one `--test` at a time (never package-wide):
  `device_fold_count_gate_test`, `device_weight_gate_test`, `device_ctr_gate_test`,
  `device_ordered_gate_test`, `device_weighted_fit_test`,
  `device_weighted_nonsym_fit_test`, `device_ctr_fit_test`, `device_ordered_fit_test`,
  `device_nonsym_fit_test`, `device_region_fit_test`, `bootstrap_dev_oracle_test`,
  `device_bootstrap_parity_test`, `device_poisson_bootstrap_test`,
  `device_one_hot_parity_test`, `device_seam_test`.
- `cargo test -p cb-backend --no-default-features --features rocm`
- `cargo test --workspace`

**Completion evidence**: all three composition assertions green; the whole device suite
green; **acceptance scenarios 6, 7, 8** closed. **D1's "all three ship together" merge
gate is satisfied only when this task is green.**

---

## 5. Coverage

### 5.1 Spec → task (all 19 GDC IDs covered)

| Spec | Task(s) | Track |
|---|---|---|
| GDC-01 | T01 | W |
| GDC-02 | T04 | W |
| GDC-03 | T05 | W |
| GDC-04 | T06 | W |
| GDC-05 | T07 | W |
| GDC-06 | T03 (weighted ×2), T13 (CTR mixed), T20 (ordered bias-0) | W / C / O |
| GDC-07 | T08 | W |
| GDC-08 | T09 | W |
| GDC-09 | T10 (types), T11 (materialization) | C |
| GDC-10 | T12 | C |
| GDC-11 | T14 | C |
| GDC-12 | T15 | C |
| GDC-13 | T02 | O |
| GDC-14 | T16 | O |
| GDC-15 | T18 (split search — BLOCKED), T19 (wiring) | O |
| GDC-16 | T21 | O |
| GDC-17 | T22 | O |
| GDC-18 | T17 | X |
| GDC-19 | T23 | X |

### 5.2 Acceptance scenario (SPEC §6) → task

| # | Scenario | Bar | Closed by |
|---|---|---|---|
| 1 | Uniform-weight fits byte-unchanged on all 4 policies | regression | T04 / T05 / T06 Verify + T07 Verify (`device_nonsym_fit_test`, `device_region_fit_test`, `session_residency`, `bootstrap_dev_oracle_test`) |
| 2 | Non-uniform-weight fit on each of the 4 policies reaches device + matches upstream | ≤1e-5 | T08 (SymmetricTree), T09 (Depthwise upstream ≤1e-5; Lossguide/Region vs CPU ε=1e-4) |
| 3 | `permutation_count>1` + CTR still declines | correctness | T14 (`multi_permutation_ctr_declines_to_device`), enabled by T01 |
| 4 | Single-permutation CTR fit reaches device, structure AND leaf values match upstream | ≤1e-5 | T15 (+ T12's unit-level discriminator) |
| 5 | Covered Ordered fit reaches device, matches upstream over all iterations | ≤1e-5 | T22 (fixture from T20) |
| 6 | Ordered × CTR still declines | correctness | T23.1 (+ T21's `ordered_plus_ctr_declines_to_device`) |
| 7 | Weighted × CTR correctly admits together | correctness | T23.2 |
| 8 | Unrelated pre-existing exclusion still excluded | regression | T23.3 |
| 9 | `begin_declines_ordered_to_cpu` reflects new behaviour | regression-intentional | T17 |

Note on row 2: T03's decision to generate two upstream scenarios (not four) means
Lossguide and Region are proved against the **CPU reference at ε=1e-4** plus the shared
weighted-formula unit oracles (T05/T06), not against a per-policy upstream fixture. This
is a deliberate, recorded planning decision with a stated escalation trigger (T03/T09).

---

## 6. Blockers, escalations, and assumptions

### BLOCKERS — require a user decision before the affected tasks execute

**Q1 (blocks T18, and therefore T19 / T21 / T22 — all of Track O's substance).**
SPEC GDC-15 says the device Ordered arm needs "the per-tree ordered trajectory apply
keyed by the tree's real `leaf_of`". Verification (**V-7**) shows:
- the CPU trainer's Ordered path differs from Plain ONLY in the *split search*
  (`greedy_tensor_search_oblivious_ordered`), and
- `ordered_approx_delta_simple` — the primitive the shipped device driver
  `accumulate_ordered_trajectory` reproduces — **is never called in the training loop**.

So the shipped ordered device driver is not on the path to upstream parity, and the real
work is a **per-segment ordered split score on the device** (`score_candidate_ordered`,
`tree.rs:2383-2418`). That is substantially larger than GDC-15's description and has no
device precedent. **Decision needed**: (a) proceed with T18 as scoped (implement the
ordered split search on the device path — largest task in the phase, likely comparable
to GDC-10); (b) re-scope Track O; or (c) defer Track O, which conflicts with D1.

**Q2 (blocks T13, and therefore T15).** Every existing CTR e2e fixture is cat-only, and
a cat-only pool yields `device_n_features == 0` / `device_n_bins == 0` so it can never
commit to the device (**V-9**); `plain_ctr/` has no trained model at all. GDC-12's
"reuse rather than regenerate" is unachievable. **Decision needed**: approve generating
a new mixed float+cat CTR fixture (T13, recipe pinned above), which is a scope addition
SPEC did not budget.

**Q3 (blocks T12's step 4).** T12's design returns the STRUCTURE-permutation `leaf_of`
on `DeviceGrownTree` while recomputing leaf VALUES over the averaging assignment. This
matches how the CPU path works (`assign_leaf_over_ctr_columns` is used only for leaf
values), but `cb_model::predict_raw`'s re-derivation of membership from splits was not
verified end-to-end this session. **Decision/verification needed before T12's Green**;
if the apply path needs the averaging assignment, this changes `DeviceGrownTree`'s
contract and must be escalated rather than silently redefined.

### CORRECTIONS to SPEC.md that the implementing agent must apply (no decision needed)

| id | SPEC says | Verified reality | Task that applies it |
|---|---|---|---|
| B-1 | GDC-01 testable with "no CTR + `permutation_count>1`" | real `fold_count` is `1` there (`fold.rs:168`); needs CTR candidates | T01 |
| — | GDC-03/04 fix point is `nonsym_grow.rs:442-448` | 3 sites in nonsym + 4 in region; ALL fixed caller-side in `session.rs:1682/1711` | T05, T06 |
| — | GDC-02 "feed weighted into hist and leaf" | `der1_h` also feeds routing (4207) and the empty-carry (4025) — needs a separate `leaf_der1_h` param | T04 |
| — | D4 "exactly one real override" | 10 mechanical edit sites (**V-5**) | T02 |
| B-2 | `body_finish`/`tail_finish` are per-tree dynamic `&[u32]` | per-FIT constants; `permutation` is `&[i32]`, boundaries are `usize` scalars (**V-6**) | T02, T19 |
| — | `DeviceCtrConfig.target_class: Vec<u8>` | actual type is `Vec<u32>`; `DeviceCtrAveraging` also needs its own `target_class` | T10 |
| B-4 | GDC-17 reuses `ordered_boost_e2e` unchanged | that fixture has `boost_from_average=true` ⇒ `bias ≈ 1.47`, blocked by the unrelaxed CR-01 clause (**V-8**) | T20 |
| B-5 | GDC-12 reuses `plain_ctr` / `ctr_btmv_simple` | `plain_ctr` has no model; all CTR fixtures are cat-only and cannot reach the device (**V-9**) | T13 |
| — | tracks are "mostly disjoint / parallelizable" | Track W's 3 kernel tasks share `grow_one`; T07/T14/T21 share ONE expression; T16/T11 share `begin` (**V-12**) | §3 waves |

### Planner decisions taken (SPEC delegated these)

- **GDC-06 scenario count** = 2 weighted upstream fixtures (SymmetricTree + Depthwise),
  with a stated escalation trigger, rather than 4 (T03).
- **GDC-09 storage shape** = a host-side `ctr_averaging_bins: Option<Vec<Vec<u32>>>` on
  `GpuTrainSession`, separate from the resident cindex — chosen because it makes
  GDC-09's "never read by the split scorer" invariant *structural* (T11).
- **GDC-08 test shape** = one `#[test]` per grow policy delegating to a shared
  `device::run(policy, label)`, mirroring `device_nonsym_fit_test.rs` (T09).
- **GDC-19 test file** = one shared `device_gate_composition_test.rs` (T23).

### Assumptions NOT verifiable in this environment

- No ROCm/CUDA hardware was available this session, so no `cargo test --features rocm`
  was executed; only `cargo check`-class verification is machine-confirmable here
  (research.md verified that `cargo check -p cb-train --no-default-features
  --features rocm --test device_nonsym_fit_test` succeeds without a GPU, and that a
  `CpuBackend`-importing test fails `E0432` under the same flags).
- Whether `session_residency_matches_cpu_multi_tree_boosting`'s existing `weight_mod5`
  scenario already fails the tightened ε=1e-4 assertion today (T04's Red) is a
  prediction from reading `LEAF_BOUND = 1e-3` (`session_residency.rs:26`), not an
  executed result.
- Upstream fixture generation (T03/T13/T20) requires a local `catboost==1.2.10`
  install; not attempted here (planning only).

### Carried residual (SPEC §9 — must appear in the completion summary)

The fixed-point histogram's `|Σ| < 2^33` precondition (`crates/cb-backend/src/kernels.rs:2318-2335`)
has **no enforced runtime guard**, and this phase does not add one. All three new
fixtures stay under it by construction with the margin documented in their generators
(T03: `64 · 3.0 · 10 = 1920` vs `8.6e9`, margin `>4.4e6×`). This is a known,
pre-existing, orthogonal limitation inherited from
`.planning/plans/device-bootstrap-parity/SPEC.md` Finding F-D, not an oversight.

---

## 7. Definition of done for the phase (D1)

All 23 tasks green, and:

```
cargo test --workspace
cargo test -p cb-backend --no-default-features --features rocm
cargo test -p cb-train --no-default-features --features rocm --test device_fold_count_gate_test
cargo test -p cb-train --no-default-features --features rocm --test device_weight_gate_test
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_gate_test
cargo test -p cb-train --no-default-features --features rocm --test device_ordered_gate_test
cargo test -p cb-train --no-default-features --features rocm --test device_weighted_fit_test
cargo test -p cb-train --no-default-features --features rocm --test device_weighted_nonsym_fit_test
cargo test -p cb-train --no-default-features --features rocm --test device_ctr_fit_test
cargo test -p cb-train --no-default-features --features rocm --test device_ordered_fit_test
cargo test -p cb-train --no-default-features --features rocm --test device_gate_composition_test
cargo test -p cb-train --no-default-features --features rocm --test device_nonsym_fit_test
cargo test -p cb-train --no-default-features --features rocm --test device_region_fit_test
cargo test -p cb-train --no-default-features --features rocm --test bootstrap_dev_oracle_test
cargo test -p cb-train --no-default-features --features rocm --test device_bootstrap_parity_test
cargo test -p cb-train --no-default-features --features rocm --test device_poisson_bootstrap_test
cargo test -p cb-train --no-default-features --features rocm --test device_one_hot_parity_test
cargo test -p cb-train --no-default-features --features rocm --test device_seam_test
```

all green on a real rocm device, and the three §6 blockers resolved.

---

## 8. COMPLETION SUMMARY (2026-08-07, executing session)

**Status: every in-scope task green on real ROCm (gfx1151); Track O deferred per D5.**
In-scope tasks executed: T01, T03–T15, T23 (T02/T16–T22 deferred with Track O).

### Results (measured, not projected)

| Oracle | Bar | Measured |
|---|---|---|
| weighted SymmetricTree e2e vs upstream (T08) | ≤1e-5 | **6.8e-9** |
| weighted Depthwise e2e vs upstream (T09) | ≤1e-5 | **7.0e-9** |
| weighted Depthwise/Lossguide/Region vs CPU ref (T09) | 1e-4 | **0.0 (exact)** |
| CTR two-permutation e2e vs upstream (T15) | ≤1e-5 | **4.5e-11** |
| weighted × CTR composition vs upstream (T23.2) | ≤1e-5 | **2.8e-17** |
| weighted-der overhead (perf probe, 200k×10 depth-6×40) | ~1.0 | **0.993** |

Full device suite: 39/39 cb-train device tests + 244/245 cb-backend self-oracles
(the 1 failure is the PRE-EXISTING `poisson_parallel_draw_outpaces` full-suite
load flake — verified by stash-bisect to fail on the unmodified baseline under
concurrent load; passes standalone every time).

### Deviations from this plan (recorded per the execution-note contract)

- **T01**: the positive `fold_count > 1` observation is asserted at the source
  (`learning_fold_count`) and end-to-end via T14's decline test, per this plan's
  own Risks note (a CTR fit cannot reach `begin_device_training` pre-T14).
- **T03/T13**: `model.json` borders are the PRUNED used-subset (verified `[2, 0]`
  on `ordered_boost_e2e`) — both weighted fixtures and the CTR fixture freeze the
  FULL quantization border set as `borders.npy` via `Pool.quantize()` +
  `save_quantization_borders` (fit-on-quantized-pool asserted bit-identical).
- **T05/T06**: the weighted oracles are SESSION-level (driving `grow_one`), not
  kernel-level — the caller-side substitution lives in the session, so a
  kernel-level test with symmetric inputs cannot discriminate.
- **T12**: simpler than planned — NO resident-approx conflict exists. The
  resident trajectory (structure-partition update) IS the CPU learning-fold-0
  approx (`UpdateLearningFold`), and the caller's host `approx` is the main
  (averaging) trajectory, so the gather reads the caller approx for its der and
  the two trajectories match the CPU CTR path by construction. The averaging
  assignment re-derives ONLY CTR-level bits from `leaf_of` + the averaging bins
  (no host cindex replay needed for the gather itself).
- **T14 grew a backend half this plan did not size**: the CPU CTR-aware search's
  `model_size_reg` cat-feature weight ((1+count/maxCount)^-0.5 on unused
  (ctr_type, projection) groups, within-tree + model-lifetime lift, phantom
  mixed float-partition maxCount contribution) had to be replicated for the
  device search — per-column scorer passes (pass C) with host-side weighting,
  group_used state on the session, and a host float-partition replay. Also fixed
  latent window wiring: CTR tail columns would previously have been scored (and
  routed!) through the one-hot equality arm.
- **Device CTR type scope NARROWED vs SPEC GDC-11**: simple **Borders** columns
  only (`target_border_idx == 0`, `prior_denom == 1`). The device kernel
  implements exactly the ordered binclf statistic; Buckets/BTMV/Counter have
  different accumulation semantics with NO device kernels, so admitting them
  would compute wrong leaves. They decline to CPU (D-04). SPEC's 4-type list
  assumed kernels the backend does not have.
- **T23.2** compares against REAL upstream weighted×CTR artifacts (additive
  frozen `gen_weighted_fixtures.py`: `weights.npy`, `borders_weighted.npy`,
  `predictions_weighted.npy`) rather than a CPU-reference proxy.
- **CTR device border table**: `borders[k] = ((k+1)·norm/bc − shift).next_down()`
  makes the device's strict `v > border` binarize EXACTLY equal to
  `calc_ctr_online_bin`'s truncation for every f64, including exact boundary
  hits (reachable at priors 0 and 1).

### Carried residual (SPEC §9 — reported as required)

The fixed-point histogram's `|Σ| < 2^33` precondition has NO enforced runtime
guard; this phase documents it at the weighted leaf-reduce call site
(`grow_oblivious_tree_resident`) and keeps every new fixture under the bound by
construction (margin > 4.4e6×, arithmetic recorded in each generator).

### External sign-off

Kaggle P100 CUDA run: kernel `yensen2/cb-rs-gdc-device-coverage-p100`
(branch `worktree-gpu-string-param-oracle` @ 78449a0) — full device suite +
perf probe under `--features cuda`; result recorded in the session summary.
