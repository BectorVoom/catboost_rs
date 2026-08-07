---
title: GPU Device Coverage Expansion (Ordered boosting / CTR / weighted derivatives)
status: draft
format: markdown
spec_version: 1
updated_at: 2026-08-07T00:00:00Z
source_requirements:
  - "User request: run GPU oracle tests for all string-valued CatBoost params, measure P100 perf where significant (session a72648e2-8f35-4634-9bed-c3cd6c59def1)"
  - "User follow-up: write up the unreachable-param findings (Ordered/CTR/class_weights excluded from device_host_eligible) as a plan"
  - "research.md: /home/user/Documents/workspace/catboost_rs/.claude/worktrees/gpu-string-param-oracle/.planning/plans/gpu-device-coverage-expansion/research.md"
---

# GPU Device Coverage Expansion — Ordered boosting, CTR categoricals, weighted derivatives

## 1. Context

An oracle-testing session (2026-08-07) found that `crates/cb-train/src/boosting.rs`'s
single per-fit device-commit gate, `device_host_eligible` (~line 4083, "GPUT-01 DEVICE
GROW SEAM", design decision D-10-01), unconditionally excludes three configurations from
ever reaching the GPU through a real `cb_train::train()` fit, even though the underlying
device kernels for each already exist and already pass their own isolated self-oracle
tests:

1. `boosting_type=Ordered` — excluded via `ordered_learning_perm.is_none()`
   (`boosting.rs:4084`). Device driver: `cb_backend::gpu_runtime::ordered::accumulate_ordered_trajectory`,
   self-oracled in `ordered_test.rs` (5/5 pass). But `GpuTrainSession::begin`
   (`session.rs:988-991`) ALSO unconditionally declines every Ordered fit before even
   consulting its own `map_ordered_coverage` result — the real blocker is architectural,
   not the `boosting.rs` clause.
2. `simple_ctr`/`combinations_ctr` (Borders/Buckets/BinarizedTargetMeanValue/Counter) —
   excluded via `materialized_ctr_features.is_empty()` /
   `structure_fold_columns.iter().all(Vec::is_empty)` (`boosting.rs:4085-4086`). Device
   CTR kernels exist and the backend gate (`ctr_covered`, `session.rs:130-144`) already
   opens real sessions — but it captures only ONE of the CPU path's TWO required
   permutations (structure vs. averaging, `boosting.rs:3680-3920`), so relaxing the gate
   naively would silently diverge from the ≤1e-5 upstream bar for leaf values.
3. `class_weights`/`auto_class_weights`/`scale_pos_weight` (any non-uniform per-object
   weight) — excluded via `weights.iter().all(|&w| w == 1.0)` (`boosting.rs:4164`,
   "WR-03"). The device histogram fill and leaf-stat reduce both consume raw `der1_h`,
   never `der1_h ⊙ weight_h`, so leaf values are `Σder1 / (Σweight + l2)` instead of
   upstream's `Σ(w·der1) / (Σweight + l2)`.

User decisions locked via `AskUserQuestion` (2026-08-07, this session — non-negotiable
per spec-tdd-planner-skill, report conflicts rather than override):

- **D1 (delivery shape): the two remaining gaps (weighted-der, CTR) ship as ONE
  combined deliverable**, not independently-mergeable waves. Internal task ORDER may
  still be sequenced for dependency correctness (gap 3 → gap 2, per `research.md`'s
  tractability ranking), but nothing merges until both are complete and green.
  **AMENDED 2026-08-07 (post-planning revision, see D5 below): gap 1 (Ordered
  boosting) is DROPPED from this phase entirely** — the Planner Agent's
  CodeGraph-verified planning pass found the premise behind gap 1's original
  "wire up the existing kernel" framing was factually wrong (D5). "All together"
  now scopes to the two gaps that remain.
- **D2 (CTR parity target): full two-permutation parity** with the CPU path (option (a)
  in `research.md`) — the device CTR arm must gather leaf-value sums from the
  AVERAGING-permutation CTR bins, distinct from the structure-permutation bins used for
  split search, matching `assign_leaf_over_ctr_columns` (`boosting.rs:5816-5817`).
- **D3 (weighted-der scope): all device grow policies** (SymmetricTree, Depthwise,
  Lossguide, Region) — not scoped to SymmetricTree only.
- **D4 (Ordered seam shape): change `Runtime::grow_tree_on_device`'s existing
  signature** directly, rather than adding a parallel seam method. Verified low-risk:
  `codegraph_explore` this session confirmed exactly ONE real override exists
  (`GpuBackend::grow_tree_on_device`, `cb-backend/src/gpu_backend.rs:310`); every other
  `Runtime` implementer (the trait default, and any `CpuRefRuntime`-style test mocks in
  `crates/cb-train/tests/device_*_test.rs`) either inherits the trait default or wraps
  `GpuBackend` (e.g. `CountingGpu` in `bootstrap_dev_oracle_test.rs`, which delegates
  every method to `self.inner`).
- **D5 (Ordered boosting DROPPED — discovered during planning, not part of the
  original research pass): the specification-planner agent's CodeGraph verification
  of GDC-15 found that the shipped device kernel
  (`cb_backend::gpu_runtime::ordered::accumulate_ordered_trajectory`, oracled against
  `ordered_approx_delta_simple`) is never actually called anywhere in the real CPU
  training loop.** The genuine difference between CPU `boosting_type=Ordered` and
  `Plain` is a per-segment SPLIT-SCORE function
  (`score_candidate_ordered`/`greedy_tensor_search_oblivious_ordered`,
  `crates/cb-train/src/tree.rs:2383-2418`), not an approx-trajectory accumulation.
  The existing "Ordered device driver" therefore does not implement what real Ordered
  boosting needs — closing this gap for real would mean designing and building a NEW
  per-segment ordered split-score device kernel from scratch, not wiring up existing
  work, making it a materially larger and differently-shaped task than any other GDC
  spec in this document. Per user decision (`AskUserQuestion`, post-planning-pass),
  **gap 1 is dropped from this phase** and left for a future, separately-researched
  phase once that kernel's real shape is sized on its own. Track O (GDC-13 through
  GDC-17) below is retained in this document for traceability but marked
  `status: deferred`, not `draft` — none of it is part of this phase's deliverable.

## 2. Scope and non-goals

### In scope
- Relaxing the TWO remaining `device_host_eligible` clauses — weighted-der and CTR —
  with correct composition guards (Ordered's clause is UNTOUCHED, per D5: it stays
  exactly as excluded as it is today).
- Extending `GpuTrainSession`/`Runtime` to actually consume CTR (both permutations)
  and non-uniform-weight configurations instead of declining or silently
  mis-computing them.
- Fixing the `fold_count` hardcoding (`boosting.rs:4297`) that would otherwise let a
  `permutation_count > 1` CTR fit silently commit to the device on fold-0 data only.
- New weighted-derivative channel plumbing in the device histogram fill and leaf-value
  kernels for the oblivious resident grow, `grow_nonsym_tree`, and the Region grower
  (D3: all three device grow paths).
- New device leaf-value-gather mechanism reading a SECOND (averaging-permutation) CTR
  cindex column set, distinct from the structure columns used for split search (D2).
- New end-to-end device-vs-CPU-and-upstream oracle tests for both remaining gaps, at
  the project's standing bars (ε=1e-4 device self-oracle per D-07; ≤1e-5 vs upstream
  CatBoost 1.2.10 for the fixture-backed oracles).
- A new weighted-training oracle fixture and a new mixed float+categorical CTR e2e
  fixture (neither exists today — confirmed during planning that NO existing CTR
  fixture is device-eligible as-is, since every existing CTR fixture is cat-only and
  `device_n_float == matrix.n_features()` requires float columns to reach the device
  at all).
- Composition guards so admitting these two does not silently also admit an
  already-excluded combination (non-SymmetricTree×bootstrap-sampling; one-hot×CTR;
  CTR×exact-leaf; CTR×sampling; and confirming Ordered stays excluded, since D5 left
  its clause untouched).

### Out of scope (unchanged from research.md, plus D5)
- **Ordered boosting on the device, entirely (D5, added during planning).** The
  premise that this was "wire up an existing kernel" was found false — the shipped
  `accumulate_ordered_trajectory` driver does not implement what real Ordered
  boosting needs (see D5 above). Deferred to a future, separately-sized phase.
- Multi-permutation CTR (`permutation_count > 1`) on the device.
- One-hot × CTR combinations on the device (SPEC-OH-26 already forbids this).
- CTR × sampling (bootstrap/MVS) and CTR × exact-leaf on the device.
- Weighted der for pairwise / ranking / multiclass device losses (already independently
  declined as forward dependencies).
- Any new external crate dependency.
- `catboost-master/` as a reference (stale 3-file stub, per top-level CLAUDE.md).
- A host-side enforced guard against the fixed-point histogram's `|Σ| < 2^33`
  overflow precondition — documented as a residual, pre-existing, orthogonal risk this
  phase inherits (also flagged in `.planning/plans/device-bootstrap-parity/SPEC.md`
  Finding F-D) rather than one it must close; the new weighted fixture (GDC-08) must
  stay comfortably under the bound and the bound must be documented at the new call
  site, but no automated enforcement is required to consider this phase complete.

## 3. Dependencies

- `crates/cb-train` → `crates/cb-backend` (via the `Runtime` trait object, behind Cargo
  features) → `crates/cb-compute` (plain-host shared types: `DeviceTrainConfig`,
  `DeviceCtrConfig`, `DeviceGrownTree`, the `Runtime` trait) → `cubecl` (workspace-pinned;
  `rocm = ["cubecl/hip"]`, `cuda = ["cubecl/cuda"]`, `default = ["cpu"]` on
  `cb-backend`).
- `cb-compute` types crossing the `cb-train`/`cb-backend` boundary MUST stay plain host
  types with no `cubecl` dependency ("T-10-04 feature-unification landmine") — any new
  field on `DeviceTrainConfig`/`DeviceCtrConfig` (e.g. the averaging-permutation CTR
  config, or the ordered per-tree state) must respect this.
- `Atomic<u64>` add support is required for the resident partition histogram fill;
  cpu/wgpu backends lack it. Every new e2e test in this plan must SKIP (print + return,
  never panic) off rocm/cuda, per the repository's existing "WR-01 anti-false-pass
  convention" (verified pattern in every precedent device test file).
- No new external crate dependencies anticipated.

## 4. Typed contracts

```rust
// crates/cb-compute/src/runtime.rs — Runtime trait, EXTENDED (D4: signature change)
trait Runtime {
    // ... compute_gradients, compute_gradients_grouped, begin_device_training unchanged ...

    /// EXTENDED (GDC-13): now additionally carries the per-tree ordered state.
    /// `ordered` is `None` for every Plain fit (byte-unchanged D-04 default path);
    /// `Some(OrderedTreeArgs)` only when the session was opened as a covered Ordered
    /// regime (GDC-14).
    fn grow_tree_on_device(
        &self,
        approx: &[f64],
        target: &[f64],
        sample: &[f64],
        ordered: Option<&OrderedTreeArgs>,   // NEW parameter, GDC-13
    ) -> CbResult<Option<DeviceGrownTree>> {
        let _ = (approx, target, sample, ordered);
        Ok(None)
    }
}

/// NEW plain-host type (GDC-13), crates/cb-compute/src/runtime.rs.
/// Carries the per-tree state accumulate_ordered_trajectory needs that the existing
/// seam does not: the FIXED (per-fit) permutation, and the PER-TREE dynamic body/tail
/// boundary (crate::fold::body_tail_boundaries, growing across iterations).
struct OrderedTreeArgs<'a> {
    permutation: &'a [u32],
    body_finish: &'a [u32],
    tail_finish: &'a [u32],
}
```

```rust
// crates/cb-compute/src/runtime.rs — DeviceCtrConfig, EXTENDED (D2: two permutations)
struct DeviceCtrConfig {
    // existing fields (structure permutation + columns) — UNCHANGED, still populated
    // from cat_learn_permutation / materialized_ctr_features.
    permutation: Vec<u32>,
    target_class: Vec<u8>,
    columns: Vec<DeviceCtrColumn>,
    /// NEW (GDC-09): the averaging permutation's OWN column materialization, used
    /// ONLY for leaf-value gather, never for split search. `None` means "not a
    /// covered CTR regime" (unchanged meaning of the outer `Option<DeviceCtrConfig>`
    /// on DeviceTrainConfig); once `Some(DeviceCtrConfig)`, this inner field is
    /// always `Some` too for a covered fit (never partially populated).
    averaging: Option<DeviceCtrAveraging>,
}

/// NEW plain-host type (GDC-09).
struct DeviceCtrAveraging {
    permutation: Vec<u32>,       // cat_averaging_permutation, distinct from `permutation` above
    columns: Vec<DeviceCtrColumn>,  // same CTR types, materialized under the averaging permutation
}
```

```rust
// crates/cb-train/src/boosting.rs — device_host_eligible, RELAXED clauses (GDC-05/11/16)
// Existing clauses NOT listed here are UNCHANGED (grow_policy, approx_dimension==1,
// !is_multiclass, !is_multilabel, bootstrap_type × grow_policy cross-product,
// random_strength==0.0, eval_sets.is_empty(), bias==0.0, leaf_method ∈ {Gradient,Simple}).
let device_host_eligible = group_spans.is_none()
    && ordered_ctr_mutually_exclusive_guard(...)          // NEW, GDC-16: !(ordered_active && has_ctr)
    // GAP 1 (was: ordered_learning_perm.is_none() — hard exclude)
    && (ordered_learning_perm.is_none() || ordered_device_covered(loss, depth, ...))  // GDC-16
    // GAP 2 (was: materialized_ctr_features.is_empty() && structure_fold_columns.iter().all(Vec::is_empty))
    && (materialized_ctr_features.is_empty()
        || (learning_folds_for_cycle == 1                  // GDC-01: fold_count fix, prerequisite
            && ctr_types_are_cpu_legal_and_device_covered(&materialized_ctr_features)))  // GDC-11
    && !penalties_active
    && params.monotone_constraints.is_empty()
    && matches!(params.grow_policy, SymmetricTree | Depthwise | Lossguide | Region)
    && approx_dimension == 1
    && !is_multiclass
    && !is_multilabel
    && (bootstrap_type==No || (bootstrap_type in {...} && grow_policy==SymmetricTree))
    && params.random_strength == 0.0
    && eval_sets.is_empty()
    && has_any_scorable_feature(&matrix)
    && one_hot_cardinalities_fit_the_device(...)
    // GAP 3 (was: weights.iter().all(|&w| w == 1.0) — hard exclude)
    // D3: no longer restricted to any single grow_policy; the weighted-der fix (GDC-02/03/04)
    // covers SymmetricTree + Depthwise/Lossguide + Region, so this clause is simply REMOVED
    // (not narrowed), subject only to the fixed-point magnitude note in Risks §9.
    && bias == 0.0
    && matches!(params.leaf_method, LeafMethod::Gradient | LeafMethod::Simple);
```

## 5. Failure-isolated behavioral specifications

Grouped by track for readability; D1 (both remaining gaps ship together) governs final
merge, not task independence — each spec below is still independently testable and has
one primary failure cause. Track O (GDC-13 through GDC-17) is retained below but
DEFERRED per D5 — it is not part of this phase's deliverable.

### Track W — Weighted derivatives (D3: all grow policies)

#### GDC-01 — Wire the real `learning_folds_for_cycle` into `begin_device_training`'s `fold_count`
- Status: draft
- Rationale: prerequisite for GDC-11 (CTR); without it, a `permutation_count > 1` CTR
  fit could silently commit to the device using only fold-0's structure columns once
  GDC-11 relaxes the CTR eligibility clause, because `ctr_covered`'s own
  `fold_count != 1` decline (`session.rs:130-144`) is currently never triggered (always
  handed the hardcoded `1`). Listed first because it is a pure bug-fix, independent of
  D1/D2/D3, and every downstream CTR spec depends on it being correct.
- Preconditions: a fit reaches `boosting.rs:4292-4313`'s `begin_device_training` call.
- Input: `learning_folds_for_cycle: usize` (already computed earlier in `train_inner`,
  per `research.md`'s CPU-side materialization section) — exact source binding to be
  confirmed by the Planner Agent via `codegraph_explore` (research.md did not pin the
  exact local variable name at the call site, only that `learning_folds_for_cycle`
  exists and is currently NOT threaded through).
- Output: `begin_device_training`'s `fold_count` argument reflects the real value, not
  the literal `1`.
- Dependencies: none (pure plumbing fix in `boosting.rs`).
- Given/When/Then:
  - Given a fit with `permutation_count > 1` and NO CTR features, when
    `begin_device_training` is called, then `fold_count` passed equals the real
    `learning_folds_for_cycle`, not `1` (assert via a unit test reading the value at the
    call site, or an integration test asserting the existing non-CTR device path is
    UNCHANGED for `permutation_count == 1`, D-04 regression guard).
  - Given a fit with `permutation_count == 1`, when `begin_device_training` is called,
    then `fold_count == 1` exactly as before (byte-unchanged regression).
- Invariants: this change alone must NOT alter `device_active` for any EXISTING
  device-eligible fit (all currently-passing device e2e tests, `permutation_count == 1`
  in every one of them per fixture inventory, must stay green unmodified).
- Acceptance tests: a new unit/integration assertion that `fold_count != 1` at
  `permutation_count > 1` (can reuse `session_ctr_gate_covers_single_permutation`'s
  existing `fold_count>1 → decline` backend assertion, now exercised from the
  `cb-train` entry point instead of only at the session level).
- Out of scope: unlocking `permutation_count > 1` CTR on the device (explicitly
  deferred, §2 Out of scope).
- Traceability: `boosting.rs:4297`; `session.rs:130-144` (`ctr_covered`).
- Unresolved: exact local variable name/scope at the `boosting.rs:4297` call site for
  `learning_folds_for_cycle` — CodeGraph-verify before task execution.

#### GDC-02 — Weighted-der channel in the oblivious resident grow (SymmetricTree)
- Status: draft
- Rationale: closes gap 3 for the SymmetricTree grow policy, the device's default and
  best-tested arm. Reuses the existing `fold_weights_resident`/`vector_mul_kernel`
  elementwise multiply — no new CubeCL kernel authored (Do Not Hand-Roll, research.md).
- Preconditions: `GpuTrainSession` has both `der1_h` and `weight_h` resident (true for
  every oblivious grow today).
- Input: `der1_h: Handle` (raw residual), `weight_h: Handle` (per-object weight, may be
  non-uniform once GDC-05 lands).
- Output: `weighted_der1_h: Handle = der1_h ⊙ weight_h`, computed once per tree.
- Dependencies: `crate::kernels::vector_mul_kernel` / `fold_weights_resident`
  (`crates/cb-backend/src/kernels/bootstrap_device.rs:618-640`) — reuse, do not fork.
- Given/When/Then:
  - Given a non-uniform `weight_h` and a `der1_h`, when the oblivious resident grow
    computes the split histogram (`launch_partition_hist2_resident_into`) and the final
    leaf-stat reduce (`launch_partition_update_into`), then BOTH read
    `weighted_der1_h`, not raw `der1_h`, for their `der1`/`score_der1` input.
  - Given a UNIFORM `weight_h` (all `1.0`), when the same code path runs, then
    `weighted_der1_h` is numerically identical to `der1_h` (regression: every existing
    uniform-weight device e2e test must stay byte-unchanged).
  - Given the NEXT tree's derivative recomputation (`launch_der_binary_resident`), when
    it reads the residual, then it uses the RAW `der1_h`, never the weighted product
    (weighting applies only to histogram/leaf-value CONTRIBUTION, not to the residual
    itself — mirrors CPU's split between `ders.der1` and `weighted_der1`,
    `boosting.rs:5152-5205`).
- Invariants: `der1_h` (raw) is never mutated in place; `weighted_der1_h` is a distinct
  handle recomputed every tree from the CURRENT `der1_h`/`weight_h`.
- Acceptance tests: new device self-oracle (unit-level, cb-backend) comparing device
  leaf VALUES against a hand-computed CPU `calc_average(Σ(w·der1), Σw, l2)` reference
  with a non-uniform weight array, at ε=1e-4 — tighter than the existing
  `session_residency_matches_cpu_multi_tree_boosting`'s generous `LEAF_BOUND`
  (research.md flagged this test as already using `weight_mod5` but not tightly
  asserting the weighted formula — confirm/tighten it as part of this spec's
  acceptance test, do not merely add a new file if the existing one can be corrected).
- Out of scope: `grow_nonsym_tree`/Region (GDC-03/GDC-04, separate specs).
- Traceability: `crates/cb-backend/src/gpu_runtime/mod.rs:4068-4089` (split histogram
  input), `mod.rs:4236-4264` (leaf-stat reduce + `calc_average` call),
  `crates/cb-backend/src/kernels/bootstrap_device.rs:618-640` (`fold_weights_resident`).
- Unresolved: none.

#### GDC-03 — Weighted-der channel in `grow_nonsym_tree` (Depthwise/Lossguide)
- Status: draft
- Rationale: D3 requires non-SymmetricTree grow policies to also honor weights; CPU
  parity target is identical to GDC-02 (`Σ(w·der1)/(Σw+l2)`), but the device
  implementation gathers `der_sub`/`w_sub` HOST-side per node (not a resident histogram
  fill), so the fix point is different code (`nonsym_grow.rs:442-448`).
- Preconditions: a fit is device-eligible with `grow_policy ∈ {Depthwise, Lossguide}`
  AND has a non-uniform weight vector (post-GDC-05).
- Input: `der_sub: &[f64]`, `w_sub: &[f64]` — the host-gathered per-node subsets before
  `calc_average`.
- Output: leaf value computed as `calc_average(sum_f64(&der_sub.iter().zip(w_sub).map(|(d,w)| d*w).collect()), sum_f64(w_sub), scaled_l2)`
  — i.e. the multiply happens on the HOST gather (this path is host-driven, unlike
  GDC-02's resident kernel), routed through `cb_core::sum_f64` (the sanctioned ordered
  summation primitive, D-07/D-08 — do not hand-roll a raw fold).
- Dependencies: `cb_core::sum_f64`.
- Given/When/Then:
  - Given a non-uniform weight vector and `grow_policy=Depthwise` (or `Lossguide`),
    when a leaf value is computed at any node, then it equals
    `Σ(w·der1)/(Σw+l2)` for that node's object subset, matching the CPU
    `leaf_wise_grower` reference within ε=1e-4.
  - Given a uniform weight vector, when the same path runs, then the result is
    numerically identical to the pre-change behavior (regression).
- Invariants: the SPLIT SCORE computation (not just leaf value) for Depthwise/Lossguide
  must also use the weighted der if the CPU reference's split score does (verify via
  CodeGraph during planning — `research.md` did not confirm whether the nonsym split
  score already reads a weighted quantity separately from the leaf value; the Planner
  Agent must check `nonsym_grow.rs`'s score-computation call sites, not just the
  leaf-value one at line 442-448, before writing the Green step).
- Acceptance tests: extend `kernels::nonsym_grow_test.rs`'s existing
  `depthwise_matches_cpu_leaf_wise_{cosine,l2}` / `lossguide_matches_cpu_leaf_wise_{cosine,l2}`
  pattern with a non-uniform-weight variant of each (4 new or parameterized cases).
- Out of scope: Region (GDC-04); SymmetricTree (GDC-02, already separate).
- Traceability: `crates/cb-backend/src/kernels/nonsym_grow.rs:442-448`.
- Unresolved: whether the nonsym split-score path has its own separate weighted-input
  requirement beyond the leaf-value site — flagged for Planner Agent CodeGraph
  verification, not assumed here.

#### GDC-04 — Weighted-der channel in the Region grower
- Status: draft
- Rationale: D3's third grow policy. Same CPU parity target and same "host-gathered,
  unweighted `calc_average`" pattern as GDC-03, but in the Region grower module.
- Preconditions: a fit is device-eligible with `grow_policy=Region` AND has a
  non-uniform weight vector (post-GDC-05).
- Input/Output: identical shape to GDC-03, applied at the Region grower's analogous
  leaf-value computation site.
- Dependencies: `cb_core::sum_f64`.
- Given/When/Then: identical structure to GDC-03, scoped to `grow_policy=Region`.
- Acceptance tests: extend `kernels::region_device_test.rs`'s
  `region_device_reproduces_frozen_cpu_region_path` pattern with a non-uniform-weight
  variant.
- Out of scope: SymmetricTree/Depthwise/Lossguide (separate specs).
- Traceability: `research.md` §"Recommended Architecture, gap 3, item 4" (exact Region
  grower file/line not yet pinned in research.md beyond "the analogous Region grower
  leaf-value computation" — Planner Agent must CodeGraph-verify the precise file/line
  before task execution, likely alongside `kernels/region_device.rs`).
- Unresolved: exact Region grower file/line for the unweighted `calc_average` call site.

#### GDC-05 — Relax `device_host_eligible`'s weight-uniformity clause
- Status: draft
- Rationale: the actual gate flip for gap 3, depends on GDC-02/03/04 landing first
  (D-04: never admit a config to the device before its weighted computation is correct
  — admitting the gate before the kernel fix would produce WRONG leaf values, worse
  than today's correct-but-CPU-only fallback).
- Preconditions: GDC-02, GDC-03, GDC-04 complete and green.
- Input: `weights: &[f64]` (already resolved by `CatBoostBuilder::resolve_weights`,
  PARAM-03, unchanged).
- Output: `device_host_eligible`'s `weights.iter().all(|&w| w == 1.0)` clause
  (`boosting.rs:4164`) is REMOVED (not narrowed to a grow_policy subset, per D3).
- Dependencies: GDC-02, GDC-03, GDC-04.
- Given/When/Then:
  - Given any non-uniform weight vector and any of the four covered grow policies, when
    `device_host_eligible` is evaluated, then it is no longer forced `false` by weight
    non-uniformity alone (other clauses still apply independently).
  - Given the SAME fit configuration as before this change except weights are now
    non-uniform, when `device_active` is checked, then it is `true` (proves the clause
    removal actually changes observable behavior, not just a dead-code deletion —
    mirrors the "assert device_active==true" pitfall-3 lesson from research.md).
- Invariants: D-10-01 (one per-fit all-or-nothing decision) preserved — no partial
  per-tree weight handling.
- Acceptance tests: GDC-08 (new e2e fixture oracle) is the primary proof; additionally
  a fast unit-level assertion that `device_host_eligible` composes correctly (does not
  newly admit an otherwise-excluded combination — see GDC-19).
- Out of scope: none beyond §2.
- Traceability: `boosting.rs:4164`.
- Unresolved: none.

#### GDC-06 — New weighted-training oracle fixture generator
- Status: draft
- Rationale: no existing fixture trains a model with non-uniform weights end-to-end
  (`crates/cb-oracle/fixtures/class_weights/` is only the raw weight ARRAY, not a
  trained model; confirmed via `grep` in research.md — no hits for
  `sample_weight=`/non-uniform-weight scenario in `gen_fixtures.py`).
- Preconditions: none (independent of code changes, can be authored in parallel with
  GDC-01..05).
- Input: a new Python generator function in `crates/cb-oracle/generator/gen_fixtures.py`
  producing an upstream `catboost==1.2.10` model trained with an explicit
  non-uniform `sample_weight` (or `class_weights`), `boost_from_average=False` (bias=0,
  matching the existing device-eligible fixture convention), `bootstrap_type=No`,
  `random_strength=0`, one representative grow policy per D3 variant needed by
  GDC-07/GDC-08 (at minimum: one SymmetricTree scenario, one non-SymmetricTree
  scenario covering Depthwise/Lossguide/Region — the Planner Agent decides exact
  scenario count against effort budget).
- Output: `X.npy`, `y.npy`, `model.json`, `predictions.npy` under a new
  `crates/cb-oracle/fixtures/weighted_device/<scenario>/` directory, matching the
  existing fixture shape (e.g. `ordered_boost_e2e/`, `plain_ctr/`).
- Dependencies: `crates/cb-oracle/generator/gen_fixtures.py`'s existing generator
  patterns (RMSE/Logloss scenario helpers already present).
- Given/When/Then:
  - Given the generator script is run, when it completes, then each new scenario
    directory contains a valid `model.json` + `predictions.npy` produced by the
    INSTALLED `catboost==1.2.10` package (not hand-computed), matching this project's
    universal "oracle-tested against installed catboost" convention.
  - Given the weight magnitudes chosen, when `n · max(weight) · max(|der1|)` is
    computed, then it stays comfortably under the fixed-point histogram's `|Σ| < 2^33`
    precondition (`kernels.rs:2318-2335`) — document the chosen bound and the margin in
    a comment in the generator script (§9 Risks: no enforced runtime guard required,
    but the fixture itself must not trip the precondition).
- Acceptance tests: the generator script runs successfully and produces
  loadable/parseable fixture files (a smoke-level Rust test loading them, mirroring
  every other `fixture()`-helper pattern in the existing `*_oracle_test.rs` files).
- Out of scope: any CTR or Ordered-boosting combination in this fixture (weighted-only,
  per §2's composition scoping).
- Traceability: `crates/cb-oracle/generator/gen_fixtures.py` (no existing weighted
  scenario); `crates/cb-oracle/fixtures/` directory conventions.
- Unresolved: exact number/shape of scenarios (SymmetricTree-only vs. one-per-grow-policy)
  — Planner Agent decides task granularity.

#### GDC-07 — E2e device oracle: weighted training, SymmetricTree
- Status: draft
- Rationale: proves GDC-02 + GDC-05 together produce a real, upstream-parity-checked
  device fit, not just an isolated kernel self-oracle.
- Preconditions: GDC-02, GDC-05, GDC-06 (SymmetricTree scenario) complete.
- Input: the GDC-06 SymmetricTree fixture.
- Output: a new `crates/cb-train/tests/device_weighted_fit_test.rs` (or similarly named,
  NOT reusing any of the ~52 `CpuBackend`-importing files — must follow the
  `device_nonsym_fit_test.rs`/`device_region_fit_test.rs` precedent: module gated
  `#[cfg(any(feature = "rocm", feature = "cuda"))]`, a hand-rolled `CpuRefRuntime` or
  direct fixture `predictions.npy` comparison, no `CpuBackend` import).
- Dependencies: `cb_train::train`, `GpuBackend::default()`, the GDC-06 fixture.
- Given/When/Then:
  - Given the fixture's non-uniform weights and `grow_policy=SymmetricTree`, when
    trained via `cb_train::train(&GpuBackend::default(), ...)`, then `device_active ==
    true` (proves the gate actually admitted it, not a silent CPU fallback — the
    research.md pitfall-3 anti-false-pass pattern).
  - Given the same fit, when predictions are computed via `cb_model::predict_raw`, then
    they match the fixture's `predictions.npy` within ≤1e-5 (the project's standard
    upstream-parity bar).
- Invariants: none beyond D-10-01/D-04 already covered by GDC-02/05.
- Acceptance tests: this spec's own new test IS the acceptance test.
- Out of scope: Depthwise/Lossguide/Region weighted e2e (GDC-08).
- Traceability: `device_nonsym_fit_test.rs`, `device_region_fit_test.rs` (structural
  precedent, full files already read in research.md).
- Unresolved: none.

#### GDC-08 — E2e device oracle: weighted training, Depthwise/Lossguide/Region
- Status: draft
- Rationale: D3 requires proof for the non-SymmetricTree grow policies too, not just
  SymmetricTree.
- Preconditions: GDC-03, GDC-04, GDC-05, GDC-06 (non-SymmetricTree scenario(s))
  complete.
- Input/Output/Given-When-Then: structurally identical to GDC-07, parameterized over
  `grow_policy ∈ {Depthwise, Lossguide, Region}` (one test function per policy, or a
  parameterized loop — Planner Agent decides).
- Dependencies: same as GDC-07 plus GDC-03/GDC-04.
- Acceptance tests: this spec's own new test(s).
- Out of scope: SymmetricTree (GDC-07).
- Traceability: same precedent files as GDC-07.
- Unresolved: none.

### Track C — CTR on device (D2: full two-permutation parity)

#### GDC-09 — Materialize the averaging-permutation CTR columns on device
- Status: draft
- Rationale: D2's core requirement. The device CTR gate today captures only the
  structure permutation; leaf values must be gathered from a SEPARATE materialization
  under `cat_averaging_permutation`, reusing the SAME online-CTR kernel a second time
  (Do Not Hand-Roll: `launch_ordered_ctr_resident` + `binarize_ctr_column_resident`,
  never fork the kernel logic).
- Preconditions: `GpuTrainSession::begin` receives a `DeviceCtrConfig` with
  `averaging: Some(DeviceCtrAveraging { permutation, columns })` populated (from
  `cb-train`, GDC-11).
- Input: `DeviceCtrAveraging.permutation: &[u32]` (the `cat_averaging_permutation`,
  distinct from the structure permutation), the same CTR type/prior configuration as
  the structure columns.
- Output: a SECOND resident binarized cindex column set (per CTR column), held
  alongside (not merged into) the structure-permutation resident cindex, addressable
  independently for leaf-value gather.
- Dependencies: `crates/cb-backend/src/kernels/ctr_device.rs`'s existing
  `launch_ordered_ctr_resident`/`binarize_ctr_column_resident`.
- Given/When/Then:
  - Given a covered CTR fit with `DeviceCtrAveraging` populated, when `begin()` runs,
    then it materializes the averaging-permutation columns via the SAME kernel called a
    second time with the averaging permutation, bit-exact vs. the CPU
    `averaging_ctr_features` reference (`boosting.rs:3906-3920`) — extends the existing
    `ctr_binarized_cindex_column_bit_exact` self-oracle pattern with a
    second-permutation case.
  - Given `DeviceCtrAveraging` is `None` (a non-CTR fit), when `begin()` runs, then
    nothing changes (regression: every existing non-CTR device path byte-unchanged).
- Invariants: the averaging-permutation columns are NEVER read by the split-search
  scorer (only by leaf-value gather, GDC-10) — this is the defining distinction from
  the structure columns.
- Acceptance tests: new self-oracle in `ctr_device_test.rs` comparing the
  averaging-permutation device columns against a CPU-transcribed
  `averaging_ctr_features` reference, bit-exact (mirrors existing CTR self-oracle
  bars).
- Out of scope: leaf-value GATHER logic (GDC-10, separate — this spec is materialization
  only).
- Traceability: `crates/cb-backend/src/gpu_runtime/session.rs:157-188`
  (`build_ctr_cindex_columns`, the function to extend/call twice),
  `crates/cb-train/src/boosting.rs:3906-3920` (CPU `averaging_ctr_features` reference).
- Unresolved: exact resident-handle storage shape for the second column set (a
  parallel `Vec<Handle>` alongside the structure one, vs. a combined struct) — Planner
  Agent's design choice within `GpuTrainSession`.

#### GDC-10 — Device leaf-value gather from averaging-permutation CTR bins
- Status: draft
- Rationale: D2's other core requirement — the device grow's `calc_average` currently
  ALWAYS reads leaf partition membership from the SAME cindex used for splitting.
  There is no existing device mechanism to gather leaf sums from a second, differently-
  binned column set; this is new logic, not a reuse of an existing kernel (research.md
  explicitly flags this as the substantial part of option (a)).
- Preconditions: GDC-09 (averaging columns materialized and resident).
- Input: the tree's finished split structure (per-object `leaf_of` under the STRUCTURE
  permutation's bins), the averaging-permutation resident columns (GDC-09).
- Output: leaf value computation re-derives per-object leaf assignment using the
  AVERAGING-permutation CTR bin value at each CTR-split node (matching CPU's
  `assign_leaf_over_ctr_columns`, `boosting.rs:5816-5817`), then computes
  `calc_average` over THAT reassignment, not the structure-permutation one.
- Dependencies: GDC-09; the existing oblivious resident leaf-stat reduce
  (`launch_partition_update_into`) as the mechanical pattern to extend (new gather step
  precedes it for CTR-split trees only).
- Given/When/Then:
  - Given a tree with at least one CTR-based split, when leaf values are computed, then
    they match the CPU reference's `assign_leaf_over_ctr_columns`-based leaf values
    within ε=1e-4 (device self-oracle bar) — this is the assertion a naive
    "structure-only" implementation would FAIL (research.md pitfall #2), so this
    Given/When/Then IS the discriminating test.
  - Given a tree with NO CTR-based splits (all float/one-hot), when leaf values are
    computed, then behavior is byte-unchanged from before this spec (regression).
- Invariants: split SEARCH always uses the structure-permutation bins (unchanged);
  only leaf-VALUE gather uses the averaging-permutation bins.
- Acceptance tests: a new self-oracle asserting leaf values on a CTR-split tree match
  the CPU `assign_leaf_over_ctr_columns` reference (not just structure-only parity) —
  this is the test research.md's pitfall #2 says a naive implementation fails.
- Out of scope: multi-fold (`permutation_count > 1`) CTR (§2 Out of scope).
- Traceability: `boosting.rs:5816-5817` (`assign_leaf_over_ctr_columns`, the CPU parity
  target); `mod.rs:4236-4264` (leaf-stat reduce, the device site to extend).
- Unresolved: exact device mechanism for "re-derive leaf assignment from a different
  column set" — no existing precedent in this codebase; Planner Agent must design this
  from scratch, flagged as the single highest-uncertainty item in this whole spec.

#### GDC-11 — Populate `DeviceTrainConfig.ctr` from cb-train and relax the CTR eligibility clauses
- Status: draft
- Rationale: the actual gate flip for gap 2, composing GDC-01 (fold_count fix) with
  GDC-09/GDC-10 (device-side two-permutation support).
- Preconditions: GDC-01, GDC-09, GDC-10 complete and green.
- Input: `materialized_ctr_features`, `cat_learn_permutation`, `averaging_ctr_features`,
  `cat_averaging_permutation` (all already computed CPU-side at `boosting.rs:3662-3920`,
  unchanged by this spec — only their DEVICE forwarding is new).
- Output: `DeviceTrainConfig.ctr = Some(DeviceCtrConfig { permutation, target_class,
  columns, averaging: Some(DeviceCtrAveraging { permutation: cat_averaging_permutation,
  columns: <materialized from averaging_ctr_features> }) })`, restricted to the 4
  CPU-legal CTR types (Borders/Buckets/BinarizedTargetMeanValue/Counter under
  `counter_calc_method=SkipTest`, matching `CPU_LEGAL_CTR_TYPES` in
  `catboost-rs-py/src/params.rs`). `device_host_eligible`'s
  `materialized_ctr_features.is_empty()` / `structure_fold_columns.iter().all(Vec::is_empty)`
  clauses (`boosting.rs:4085-4086`) are relaxed, guarded by
  `learning_folds_for_cycle == 1` (GDC-01), `one_hot_bins.is_empty()` (already
  structurally implied), and the existing sibling clauses (no penalties, no monotone
  constraints, no eval_sets — unchanged).
- Dependencies: GDC-01, GDC-09, GDC-10.
- Given/When/Then:
  - Given a single-permutation CTR fit using only CPU-legal CTR types, when
    `device_host_eligible` is evaluated, then it is no longer forced `false` by CTR
    presence alone.
  - Given the SAME CTR fit, when `device_active` is checked, then it is `true` (proves
    observable behavior change, not a no-op — mirrors research.md pitfall #3's lesson
    applied here).
  - Given a `permutation_count > 1` CTR fit, when `device_host_eligible` is evaluated,
    then it STAYS `false` (the `learning_folds_for_cycle == 1` guard fires — regression
    proof that GDC-01 actually prevents the silent-fold-0 bug).
  - Given a CTR fit using `FloatTargetMeanValue`/`FeatureFreq` (GPU-only-upstream types
    already rejected CPU-side by `catboost-rs-py`), when reached, then it is
    unreachable by construction (the Python layer already rejects these before any
    fit — this Given/When/Then documents the invariant, no new guard code needed here).
- Invariants: D-10-01 preserved (all-or-nothing per fit).
- Acceptance tests: GDC-12 (e2e fixture oracle) is the primary proof; plus a fast
  unit-level `permutation_count > 1` decline assertion (mirrors
  `session_ctr_gate_covers_single_permutation`, exercised from `cb-train`).
- Out of scope: multi-fold CTR (§2).
- Traceability: `boosting.rs:4085-4086, 4226-4291, 4297`.
- Unresolved: none beyond GDC-01's variable-name unresolved item.

#### GDC-12 — E2e device oracle: CTR training, structure AND leaf values
- Status: draft
- Rationale: proves the FULL two-permutation implementation (GDC-09+10+11), not just
  split structure — research.md pitfall #2 explicitly warns a structure-only test would
  pass even with a wrong (structure-only) leaf-value implementation, so this spec's
  acceptance bar MUST check leaf values / final predictions, not just tree structure.
- Preconditions: GDC-11 complete.
- Input: existing fixtures `crates/cb-oracle/fixtures/plain_ctr/` (Borders, single cat
  column, Logloss, `permutation_count=1`) and/or
  `crates/cb-oracle/fixtures/ctr_btmv_simple/` (BinarizedTargetMeanValue, cat-only,
  Logloss, `permutation_count=1`, already ships `model.cbm` + `predictions.npy`) — both
  already pin the device-covered regime per research.md; reuse rather than regenerate,
  BUT first confirm (per research.md's own caveat) that the fixture's structure and
  averaging permutations do NOT coincidentally agree at its specific `n`/seed (if they
  do, the fixture would not discriminate a structure-only bug from a correct
  implementation — Planner Agent must verify this before treating either fixture as
  sufficient).
- Output: a new `crates/cb-train/tests/device_ctr_fit_test.rs`, following the same
  `#[cfg(any(feature = "rocm", feature = "cuda"))]` / no-`CpuBackend`-import precedent
  as GDC-07/GDC-08.
- Dependencies: `cb_train::train`, `GpuBackend::default()`, the reused fixture(s).
- Given/When/Then:
  - Given the fixture's CTR configuration, when trained via
    `cb_train::train(&GpuBackend::default(), ...)`, then `device_active == true`.
  - Given the same fit, when predictions are computed, then they match the fixture's
    `predictions.npy` within ≤1e-5 — this bar is the one a structure-only
    implementation FAILS (per pitfall #2), making this the discriminating acceptance
    test for D2's "full parity" decision.
- Invariants: none beyond GDC-11.
- Acceptance tests: this spec's own new test IS the acceptance test.
- Out of scope: any fixture requiring `permutation_count > 1` or a GPU-only CTR type.
- Traceability: `crates/cb-oracle/fixtures/plain_ctr/`, `crates/cb-oracle/fixtures/ctr_btmv_simple/`.
- Unresolved: whether the reused fixtures' structure/averaging permutations actually
  diverge at their pinned seed (must verify, not assume) — if they coincidentally
  agree, a NEW fixture must be generated instead (extend GDC-06's generator work).

### Track O — Ordered boosting on device — **DEFERRED, NOT PART OF THIS PHASE (D5)**

> Every spec in this track is retained for traceability only. During planning,
> CodeGraph verification of GDC-15 found the premise wrong: the shipped
> `accumulate_ordered_trajectory` device kernel does not implement what CPU Ordered
> boosting actually does (the real difference from Plain is a per-segment split-score
> function, `score_candidate_ordered`, not an approx-trajectory accumulation). Closing
> this gap for real needs a NEW device kernel designed from scratch — out of scope for
> this phase per user decision. None of GDC-13 through GDC-17 below has a PLAN.md task;
> do not implement them as written. A future phase should re-research Ordered device
> support from `score_candidate_ordered`/`greedy_tensor_search_oblivious_ordered`
> (`crates/cb-train/src/tree.rs:2383-2418`), not from these specs.

#### GDC-13 — Extend `Runtime::grow_tree_on_device`'s signature
- Status: deferred (D5 — Track O out of scope for this phase; retained for traceability only)
- Rationale: D4's locked decision. Verified via `codegraph_explore` this session:
  exactly one real override exists (`GpuBackend::grow_tree_on_device`,
  `gpu_backend.rs:310`); the trait default and every test mock (e.g. `CountingGpu` in
  `bootstrap_dev_oracle_test.rs`, which delegates to `self.inner`) are unaffected by a
  signature change beyond adding one parameter with a sensible default-forwarding
  pattern.
- Preconditions: none (pure trait/type change, can start immediately, does not depend
  on Track W or C).
- Input/Output: see §4 Typed contracts (`OrderedTreeArgs<'a>`, the new
  `ordered: Option<&OrderedTreeArgs>` parameter).
- Dependencies: none beyond the trait definition itself.
- Given/When/Then:
  - Given the trait default implementation, when `ordered` is passed as `Some(...)` or
    `None`, then the default still returns `Ok(None)` unconditionally (byte-unchanged
    default behavior — the new parameter is bound-and-ignored in the default, exactly
    like every other parameter today).
  - Given `GpuBackend`'s override, when called with `ordered: None` (every currently-
    passing test today), then behavior is byte-unchanged (regression proof).
  - Given every OTHER `Runtime` call site in the codebase (the ONE caller in
    `boosting.rs`, plus any test-local `Runtime` impls), when this signature changes,
    then the crate compiles cleanly with the new parameter threaded through as `None`
    everywhere except the Ordered-specific call site added in GDC-15.
- Invariants: no behavior change for any NON-Ordered fit from this spec alone.
- Acceptance tests: `cargo check`/`cargo test` across the workspace compiles and every
  PRE-EXISTING test stays green (this spec's primary risk is a compile-time regression
  across ~2 call sites, not a logic bug — the acceptance bar is "nothing broke").
- Out of scope: actually USING the new parameter for real ordered accumulation
  (GDC-15).
- Traceability: `crates/cb-compute/src/runtime.rs:1376-1384` (trait method to extend),
  `crates/cb-backend/src/gpu_backend.rs:310-335` (the one real override).
- Unresolved: none — this is the lowest-uncertainty spec in the whole plan.

#### GDC-14 — Consume `map_ordered_coverage`'s result instead of discarding it
- Status: deferred (D5 — Track O out of scope for this phase; retained for traceability only)
- Rationale: the actual architectural blocker research.md identified — `begin()`
  currently computes `map_ordered_coverage(...)` and immediately discards it, returning
  `Ok(None)` unconditionally for any `!boosting_type_is_plain` fit
  (`session.rs:988-991`). This spec makes the discarded result load-bearing.
- Preconditions: none beyond the existing `map_ordered_coverage`/`OrderedState` code
  already present (`#[allow(dead_code)]` today).
- Input: `map_ordered_coverage(loss, config, depth, fold_count) -> Option<OrderedState>`
  (existing function, unchanged signature).
- Output: when `Some(OrderedState)`, `begin()` proceeds to construct a covered session
  (storing the `OrderedState` in the `ordered` field, no longer always `None`); when
  `None`, `begin()` still declines to `Ok(None)` as today.
- Dependencies: none beyond existing `map_ordered_coverage`.
- Given/When/Then:
  - Given a loss/depth/config combination `map_ordered_coverage` classifies as covered,
    when `begin()` is called with `boosting_type_is_plain == false`, then it does NOT
    immediately return `Ok(None)` — it proceeds to session construction (the remaining
    gate clauses in `begin()`, e.g. depth/loss/score-function coverage, still apply
    identically to the Plain path).
  - Given a combination `map_ordered_coverage` classifies as NOT covered, when
    `begin()` is called, then it still declines to `Ok(None)` (unchanged for the
    uncovered subset).
  - Given the EXISTING `begin_declines_ordered_to_cpu` self-test
    (`ordered_test.rs:250-313`), when this spec lands, then that test's assertions must
    be UPDATED (not silently left failing) to reflect that a covered ordered
    configuration NO LONGER always declines — this is an intentional, expected test-
    update site (research.md's own "Verification Only" note), not a regression to
    paper over.
- Invariants: the remaining `begin()` gate clauses (depth, loss, score function,
  bootstrap, etc.) still apply to the Ordered path exactly as they do to Plain — no new
  cross-family exemption.
- Acceptance tests: an updated `ordered_test.rs` where a covered-config test now
  asserts session construction SUCCEEDS (paired with a still-declines test for an
  uncovered config, so both directions stay tested).
- Out of scope: the actual per-tree grow wiring (GDC-15).
- Traceability: `crates/cb-backend/src/gpu_runtime/session.rs:974-991` (the
  unconditional early return to remove), `session.rs:524-566` (`OrderedState`,
  `map_ordered_coverage`).
- Unresolved: none.

#### GDC-15 — Wire the per-tree ordered trajectory apply using the tree's real `leaf_of`
- Status: deferred (D5 — the CodeGraph verification that found this spec's premise
  factually wrong was performed AGAINST this spec; retained verbatim for traceability
  of what was found wrong, not as a task to execute)
- Rationale: the existing `accumulate_ordered_trajectory` self-oracle
  (`ordered_test.rs`) uses an IDENTITY leaf map because it tests the accumulation
  PRIMITIVE in isolation. A real per-tree grow needs the ACTUAL `leaf_of` from that
  tree's structure search, which this spec supplies. Reuses
  `launch_apply_leaf_delta_into` (Do Not Hand-Roll — already used by the oblivious
  resident grow's own approx update).
- Preconditions: GDC-13 (seam signature), GDC-14 (session admits covered ordered
  configs).
- Input: `OrderedTreeArgs` (GDC-13's new type: permutation, body_finish, tail_finish)
  PLUS the current tree's real per-object `leaf_of` (produced by that tree's structure
  search, same as the Plain path already produces).
- Output: `grow_tree_on_device`'s Ordered arm applies the per-tree trajectory delta
  keyed by the REAL `leaf_of`, via `launch_apply_leaf_delta_into`, then reads back the
  updated resident trajectory exactly once per tree (matching the existing residency
  discipline).
- Dependencies: GDC-13, GDC-14, `launch_apply_leaf_delta_into`
  (`cb_backend::gpu_runtime`, existing kernel).
- Given/When/Then:
  - Given a covered Ordered fit growing its Nth tree, when `grow_tree_on_device` is
    called with `Some(OrderedTreeArgs)`, then the device trajectory update uses that
    tree's REAL `leaf_of` (not an identity map), and the resulting per-object approx
    trajectory matches the CPU `ordered_approx_delta` reference within ε=1e-4 (extends
    `ordered_test.rs`'s existing `device_trajectory_matches_frozen_cpu_reference`
    pattern from an isolated-primitive test to a real multi-tree grow scenario).
  - Given the body/tail boundary sequence growing across iterations
    (`crate::fold::body_tail_boundaries`), when multiple trees are grown in sequence,
    then each tree's `body_finish`/`tail_finish` reflects the CORRECT per-iteration
    boundary (not a stale or fixed one) — this is the per-tree DYNAMIC state research.md
    identified as genuinely new (not carried by any existing seam).
- Invariants: D-05 (only O(1) per-level decisions cross the seam; bulk data stays
  resident) — the trajectory read-back happens ONCE per tree (or once total, matching
  the existing `accumulate_ordered_trajectory` residency pattern), never per-object per
  level.
- Acceptance tests: GDC-17 (e2e fixture oracle) is the primary end-to-end proof; a
  tightened unit-level extension of `ordered_test.rs`'s trajectory test using a REAL
  (non-identity) `leaf_of` sequence is the primary unit-level proof.
- Out of scope: the `boosting.rs` clause relaxation itself (GDC-16).
- Traceability: `crates/cb-backend/src/gpu_runtime/ordered.rs`
  (`accumulate_ordered_trajectory`, `ordered_approx_delta`, `OrderedTree`),
  `crate::fold::body_tail_boundaries` (the per-iteration boundary source, `cb-train`
  side — confirm exact cross-crate accessibility via CodeGraph, since `cb-backend`
  cannot depend on `cb-train`, T-10-04 landmine; the boundary values must be PASSED IN
  via `OrderedTreeArgs`, never computed inside `cb-backend`).
- Unresolved: confirm `body_tail_boundaries` is computable entirely `cb-train`-side and
  passed as plain `&[u32]` slices through `OrderedTreeArgs` (§4 already assumes this;
  Planner Agent must verify no `cb-train` type leaks across the seam, only primitive
  slices).

#### GDC-16 — Relax `device_host_eligible`'s Ordered clause with the Ordered×CTR composition guard
- Status: deferred (D5 — Track O out of scope for this phase; retained for traceability only)
- Rationale: the actual gate flip for gap 1, PLUS the explicit composition guard
  research.md's pitfall #7 requires: CPU already treats Ordered×CTR as mutually
  exclusive (`boosting.rs:5443-5444`); this expansion must not let the union of GDC-11
  (CTR relax) and this spec (Ordered relax) accidentally admit a combined Ordered+CTR
  fit that no kernel (device or CPU) actually implements together.
- Preconditions: GDC-13, GDC-14, GDC-15 complete and green. (Also benefits from GDC-11
  existing, for the composition guard to have something to guard against — but does not
  strictly require GDC-11 to be MERGED first, since the guard can be written
  defensively either order; Planner Agent sequences per D1's "ship together" final
  merge.)
- Input: `ordered_learning_perm: Option<...>` (existing), `materialized_ctr_features`
  (existing, shared with GDC-11).
- Output: `device_host_eligible`'s `ordered_learning_perm.is_none()` clause
  (`boosting.rs:4084`) is relaxed to admit a covered Ordered fit, GUARDED by an explicit
  `!(ordered_active && has_ctr)` (or equivalent) so the two relaxations never compose
  into an admitted combination.
- Dependencies: GDC-13, GDC-14, GDC-15.
- Given/When/Then:
  - Given a covered `boosting_type=Ordered` fit with NO CTR features, when
    `device_host_eligible` is evaluated, then it is no longer forced `false` by
    `ordered_learning_perm.is_some()` alone.
  - Given a fit with BOTH `boosting_type=Ordered` AND CTR-routed categoricals, when
    `device_host_eligible` is evaluated, then it STAYS `false` (or the fit takes the
    CPU path, matching the existing CPU-side mutual exclusion) — this is the negative
    test research.md's pitfall #7 explicitly demands.
- Invariants: D-10-01 preserved.
- Acceptance tests: GDC-17 (positive e2e) plus GDC-19's Ordered×CTR negative
  composition test.
- Out of scope: none beyond §2.
- Traceability: `boosting.rs:4084`, `boosting.rs:5443-5444` (CPU mutual-exclusion
  reference).
- Unresolved: none.

#### GDC-17 — E2e device oracle: Ordered boosting training
- Status: deferred (D5 — Track O out of scope for this phase; retained for traceability only.
  Also note: planning found `ordered_boost_e2e`'s existing fixture is NOT device-eligible
  as-is anyway, boost_from_average=true, so this spec's "no new fixture needed" premise
  was independently wrong too — a future Ordered phase needs a new fixture regardless.)
- Rationale: the ONE case among the three gaps with a directly reusable existing
  fixture (`ordered_boost_e2e`, already frozen, already ≤1e-5-verified on CPU) — no new
  fixture generation needed.
- Preconditions: GDC-16 complete.
- Input: `crates/cb-oracle/fixtures/ordered_boost_e2e/` (existing: `X.npy`, `y.npy`,
  `model.json`, `predictions.npy`).
- Output: a new `crates/cb-train/tests/device_ordered_fit_test.rs` (or a
  `GpuBackend`-driven variant added alongside the existing CPU
  `ordered_boost_e2e_oracle_test.rs`, but as a SEPARATE file per the no-`CpuBackend`-
  import precedent — `ordered_boost_e2e_oracle_test.rs` itself imports `CpuBackend` and
  would fail to compile under `--no-default-features`, confirmed by research.md's
  `cargo check` reproduction).
- Dependencies: `cb_train::train`, `GpuBackend::default()`, the existing fixture.
- Given/When/Then:
  - Given the `ordered_boost_e2e` fixture, when trained via
    `cb_train::train(&GpuBackend::default(), ...)`, then `device_active == true`.
  - Given the same fit, when predictions are computed, then they match
    `predictions.npy` within ≤1e-5, over ALL 5 iterations (matching the existing CPU
    oracle's exact bar, `crates/cb-train/tests/ordered_boost_e2e_oracle_test.rs`).
- Invariants: the EXISTING CPU `ordered_boost_e2e_oracle_test.rs` stays green,
  untouched (this spec adds a device SIBLING, never modifies the CPU oracle).
- Acceptance tests: this spec's own new test IS the acceptance test.
- Out of scope: none.
- Traceability: `crates/cb-train/tests/ordered_boost_e2e_oracle_test.rs` (fixture +
  bar to mirror).
- Unresolved: none.

### Track X — Cross-gap composition and regression guards

#### GDC-18 — Update `begin_declines_ordered_to_cpu`'s pinned expectations
- Status: deferred (D5 — depends entirely on GDC-14, which is deferred; the existing
  test's "always declines" assertion stays CORRECT and unchanged for this phase, since
  the Ordered clause is untouched — nothing to update)
- Rationale: GDC-14 makes this existing test's current assertion ("a covered ordered
  RMSE declines to CPU pending the ordered grow seam") FALSE by design. Research.md
  explicitly flags this as an intentional, expected update site, not a regression —
  but it must be tracked as its OWN spec so it is not silently forgotten or silently
  "fixed" by weakening the assertion instead of correctly flipping it.
- Preconditions: GDC-14.
- Input/Output: the existing test file `crates/cb-backend/src/gpu_runtime/ordered_test.rs`,
  specifically the `begin_declines_ordered_to_cpu` function.
- Dependencies: GDC-14.
- Given/When/Then:
  - Given the SAME covered-config input this test used before, when GDC-14 lands, then
    the test is updated to assert session construction SUCCEEDS (not declines) for that
    input, AND a NEW, separate assertion (same test or a new one) confirms an
    UNCOVERED config still correctly declines — the negative case must not be lost in
    the process of fixing the positive one.
- Invariants: none beyond faithfully reflecting GDC-14's new behavior.
- Acceptance tests: the updated test itself, green.
- Out of scope: none.
- Traceability: `crates/cb-backend/src/gpu_runtime/ordered_test.rs:250-313`.
- Unresolved: none.

#### GDC-19 — Cross-gap composition regression tests
- Status: draft
- Rationale: the general D-10-01 concern (no gap should silently widen an UNRELATED
  existing exclusion) needs explicit, independently-checkable tests, not just inline
  guards trusted by inspection. **Amended for D5**: since Track O is dropped and its
  `device_host_eligible` clause is left completely untouched, "Ordered + CTR still
  declines" is now a plain, lower-stakes REGRESSION check (the clause literally did
  not change) rather than a guard against two simultaneous relaxations interacting —
  kept anyway because it is cheap and directly verifies D5's "untouched" claim rather
  than merely asserting it in prose.
- Preconditions: GDC-05, GDC-11 (both gate relaxations landed).
- Input: synthetic fit configurations combining two-or-more of {non-uniform weights,
  CTR features, `boosting_type=Ordered`} plus at least one ALREADY-excluded combination
  from before this phase (e.g. non-SymmetricTree + non-`No` bootstrap_type, still
  excluded per the untouched cross-product clause).
- Output: a set of assertions (can live in a shared new test file, e.g.
  `crates/cb-train/tests/device_gate_composition_test.rs`) proving:
  1. `boosting_type=Ordered` (alone, or combined with CTR features) still declines to
     CPU exactly as it did before this phase (regression proof that GDC-05/GDC-11 did
     not accidentally touch the untouched Ordered clause).
  2. Weighted + CTR together correctly ADMITS to device (research.md explicitly says
     this combination IS plausible/intended — the CTR-augmented cindex columns are just
     additional feature columns the weighted histogram fill reads like any other,
     confirmed via `session_ctr_augments_resident_cindex`), i.e. this is a POSITIVE
     composition test, not a negative one — must not be conflated with #1.
  3. An already-excluded combination UNRELATED to this phase (e.g. Depthwise +
     `bootstrap_type=Bayesian`, excluded since before this phase per
     `boosting.rs:4118-4125`) is STILL excluded after both gaps land (proves no gap
     accidentally widened an unrelated clause).
- Dependencies: GDC-05, GDC-11.
- Given/When/Then: see Output above, one Given/When/Then per numbered item.
- Invariants: this spec is purely verification — it must not require any NEW
  production code beyond what GDC-05/11/16 already implement; if any assertion here
  fails, the fix belongs in GDC-05/11/16, not in this spec.
- Acceptance tests: this spec's own new test file.
- Out of scope: none.
- Traceability: `boosting.rs:5443-5444` (Ordered×CTR CPU exclusion),
  `session_ctr_augments_resident_cindex` (session_depth_gt1_test.rs, precedent for
  item 2), `boosting.rs:4118-4125` (the untouched bootstrap×grow_policy cross-product,
  precedent for item 3).
- Unresolved: none.

## 6. Acceptance scenarios (summary)

Rows 5, 6, and 9 from the original draft (Ordered-specific) are REMOVED per D5 — Track
O is deferred, not part of this phase's acceptance bar. Renumbered:

| # | Scenario | Bar | Spec(s) |
|---|---|---|---|
| 1 | Uniform-weight SymmetricTree/Depthwise/Lossguide/Region fits behave byte-unchanged | regression | GDC-02,03,04,05 |
| 2 | Non-uniform-weight fit on each of the 4 grow policies reaches device and matches upstream | ≤1e-5 | GDC-07, GDC-08 |
| 3 | `permutation_count>1` + CTR still declines to CPU | correctness | GDC-01, GDC-11 |
| 4 | Single-permutation CPU-legal CTR fit reaches device, structure AND leaf values match upstream | ≤1e-5 | GDC-12 |
| 5 | `boosting_type=Ordered` (alone or + CTR) still declines to CPU, UNCHANGED from before this phase | regression | GDC-19.1 |
| 6 | Weighted × CTR correctly admits together | correctness | GDC-19.2 |
| 7 | An unrelated pre-existing exclusion (e.g. Depthwise+Bayesian) still excluded | regression | GDC-19.3 |

## 7. Impact scope

- **local**: kernel-level changes inside `cb-backend` (histogram fill input handles,
  new CTR averaging materialization) — contained within
  `crates/cb-backend/src/gpu_runtime/` and `crates/cb-backend/src/kernels/`.
- **cross-module**: `crates/cb-train/src/boosting.rs` (the gate + `DeviceTrainConfig`
  construction) ↔ `crates/cb-backend` (session/kernels) ↔ `crates/cb-compute`
  (`DeviceTrainConfig`/`DeviceCtrConfig` shared types). Per D5, the `Runtime` trait
  itself is NOT touched by this phase — `Runtime::grow_tree_on_device`'s signature
  stays exactly as it is today; that change was scoped entirely to the now-deferred
  Track O (GDC-13) and does not apply to the weighted-der/CTR work.
- **cross-service**: none (single Rust workspace, no network/service boundary).
- **external/public**: none directly — `catboost-rs-py`'s Python surface is
  unaffected (no `task_type` param exists yet at that layer; this phase is entirely
  internal to the Rust engine's device dispatch, reachable only via
  `cb_train::train(&GpuBackend, ...)` called directly, same as today).
- **operational**: new test commands (`cargo test -p cb-train --no-default-features
  --features rocm --test <name>`, one per new e2e file) must be added to whatever CI/
  local-verification checklist the project uses for GPU work; no deployment/secrets
  impact.

## 8. Compatibility and migration

- No data migration; no on-disk model format change (the CTR device path must produce
  predictions numerically compatible with existing model files — tree structures are
  unchanged in shape, only which BACKEND grows them, and only WHICH derivative values
  feed the same leaf-value formula for the weighted case).
- No `Runtime` trait signature changes in this phase (D5 — that was Track O/GDC-13
  only, now deferred). No breaking changes to any `Runtime` implementer.
- No public Python API surface changes (confirmed: `task_type` remains `KnownNotYet` in
  `catboost-rs-py/src/params.rs`; out of scope for this phase entirely).

## 9. Risks and open questions

Carried forward from `research.md`, not re-litigated here (already resolved by user
decisions D1-D4) except where a residual risk remains after those decisions:

- **Residual (documented, not closed by this phase)**: the fixed-point histogram's
  `|Σ| < 2^33` overflow precondition has NO enforced runtime guard anywhere in the
  codebase today (confirmed also true for the pre-existing bootstrap-sample-multiplier
  case, `.planning/plans/device-bootstrap-parity/SPEC.md` Finding F-D). GDC-06's new
  weighted fixture must stay comfortably under this bound by construction and document
  the margin; no new enforced guard is required for this phase to be considered
  complete, but this is a known limitation, not an oversight, and should be called out
  in the PLAN.md's completion summary.
- **Unresolved implementation-detail items** (each already flagged inline in its
  owning spec above, listed here for visibility): GDC-01's exact
  `learning_folds_for_cycle` variable binding; GDC-03's nonsym split-score weighted-input
  question; GDC-04's exact Region grower file/line; GDC-06's exact fixture scenario
  count; GDC-09's resident-handle storage shape; GDC-12's fixture-permutation-divergence
  verification; GDC-15's `body_tail_boundaries` cross-crate-accessibility confirmation.
  None of these block SPEC.md from being considered complete — they are exactly the
  class of detail the Planner Agent resolves via CodeGraph during PLAN.md authoring,
  per the skill's own workflow (`SPEC.md` defines WHAT/WHY, `PLAN.md` defines the
  verified HOW).
- **Effort note (not a spec gap, a planning-visibility note)**: this phase is, by the
  user's own explicit choices (D1 "all together", D2 "full parity", D3 "all grow
  policies", D4 "signature change"), the MAXIMAL-scope combination research.md
  presented — GDC-10 (device leaf-value gather from a second permutation) in
  particular has no existing precedent anywhere in this codebase and should be
  expected to be the single largest, highest-uncertainty task in the resulting
  PLAN.md.

## 10. Traceability and sources

- `[VERIFIED: LOCAL research.md]` — the complete phase-research-agent report at
  `/home/user/Documents/workspace/catboost_rs/.claude/worktrees/gpu-string-param-oracle/.planning/plans/gpu-device-coverage-expansion/research.md`,
  itself grounded in `codegraph_explore` queries and local `Read`/`Bash` verification
  (full source list in that document's own §Sources).
- `[VERIFIED: CODEGRAPH crates/cb-compute/src/runtime.rs:Runtime,grow_tree_on_device]`
  and `[VERIFIED: CODEGRAPH crates/cb-backend/src/gpu_backend.rs:GpuBackend]` — this
  session's own follow-up `codegraph_explore` confirming exactly one real
  `grow_tree_on_device` override exists, grounding D4's "low-risk" framing.
  `[VERIFIED: LOCAL /home/user/.claude/projects/-home-user-Documents-workspace-catboost-rs/memory/gpu-string-param-oracle-coverage-2026-08-07.md]`
  — this session's own prior memory write, the origin of the three-gap finding.
- User decisions D1-D4: `[VERIFIED: LOCAL this conversation]` — captured via
  `AskUserQuestion` in this session, non-negotiable per skill instructions.
