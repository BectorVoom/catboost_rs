# Phase Research: GPU Full Parameter Parity (device-eligibility gap closure + speed-supremacy benchmark)

## Research Summary

- **Phase goal.** Two independent deliverables requested by the user: (1) an exhaustive,
  file:line-precise survey of every remaining `device_host_eligible` /
  `GpuTrainSession::begin` exclusion clause in `crates/cb-train/src/boosting.rs` and
  `crates/cb-backend/src/gpu_runtime/session.rs`, classified by whether a device kernel
  already exists for it; (2) a methodology assessment for a rigorous, non-cherry-picked
  "catboost_rs vs official CatBoost GPU" Kaggle P100 speed benchmark.
- **Recommended approach.** Do NOT treat this as one uniform relaxation. The gaps split
  into three materially different shapes that a planner must size separately:
  1. **Pure wiring (kernel + coverage gate landed, per-tree grow seam never built)** —
     pairwise (`*Pairwise`), ranking (QueryRMSE/QuerySoftMax/YetiRank), multiclass/
     multi-output (MultiClass/MultiClassOneVsAll/MultiLogloss/MultiCrossEntropy/
     RMSEWithUncertainty), and **exact-leaf (MAE/Quantile/MAPE)** — a
     previously-unflagged, HIGH-confidence finding of this research pass. All four
     require extending `Runtime::grow_tree_on_device`'s `(approx, target, sample)`
     signature to carry a per-family descriptor (query grouping, per-object pair/group,
     K-dim block leaf, or nothing extra for exact-leaf — its gate is already fully wired,
     it is simply never *allowed* to reach `begin()` from `cb-train`).
  2. **Genuinely new device kernel design required** — Ordered boosting (confirmed,
     unchanged since the prior phase's D5 finding: the shipped
     `accumulate_ordered_trajectory` driver does not implement what CPU Ordered actually
     does; the real target is a per-segment ordered split-score kernel mirroring
     `score_candidate_ordered`), the other three CTR accumulation semantics (Buckets /
     BinarizedTargetMeanValue / Counter), monotone_constraints, and feature-weight
     penalties. None of these have ANY device-side code today (verified by exhaustive
     grep — zero non-coincidental hits).
  3. **CPU-side feature does not exist at all** — Langevin/SGLB
     (`langevin`/`diffusion_temperature`/`posterior_sampling`). The device
     `AddLangevinNoise` kernel is self-oracled in isolation, but there is **no**
     `BoostParams` field, no `train_inner` code path, and the Python params registry
     marks all three `KnownNotYet`. Device wiring is meaningless until a CPU parity
     target exists — this is out of GPU-phase scope entirely and should be flagged to
     the user as a different kind of gap (a CPU feature gap, not a device-porting gap).
  4. **Small, well-scoped fixes** — `boost_from_average=true` / nonzero bias (CR-01: the
     session hardcodes a zero-init resident approx at
     `crates/cb-backend/src/gpu_runtime/session.rs:1580`; passing the real bias through
     is a signature extension, not an architecture change) and non-symmetric-grow ×
     bootstrap sampling (the same `weighted_der1`-shaped fix the prior phase just shipped
     for weights applies almost verbatim to a host-computed sample multiplier).
  5. **Deliberately out of reach / composition guards that must stay** — one-hot × CTR
     (SPEC-OH-26), CTR × sampling, CTR × exact-leaf, multi-permutation CTR, MVS without a
     caller-pinned λ (iter-0 auto-λ, MVP-deferred), Newton leaf on Logloss/CrossEntropy
     (CR-02, deliberately excluded — numerically would diverge from the device's
     Gradient/`calc_average` leaf formula).
  6. **Architectural tension the planner must resolve before scoping "task_type"**: this
     project's backend selection is **compile-time only** (Cargo feature flag; CLAUDE.md
     constraint, verified in `crates/catboost-rs/src/builder.rs:1187-1190` — the backend
     binding is chosen by `#[cfg(feature = "cpu")]` / `#[cfg(any(feature = "wgpu",
     "cuda", "rocm"))]`, not a runtime branch). A GPU-feature wheel already attempts
     device dispatch for every eligible fit with **no** `task_type` kwarg needed; a
     CPU-feature wheel can never honor `task_type="GPU"` regardless of the kwarg value.
     Adding `task_type` as an IMPLEMENTED Python param therefore cannot mean "runtime
     backend switch" (forbidden by this project's own constraints) — at most it could
     mean an informational/consistency-checked no-op. This must be an explicit user
     decision before any implementation, not something the planner infers silently.
- **Most important constraints.**
  - D-10-01 (per-fit, all-or-nothing device commitment) and D-04 (byte-unchanged CPU
    path for anything not admitted) still govern every clause; nothing here should be
    planned as a partial/per-tree fallback.
  - GPU tests only run for real on `rocm`/`cuda`; this repo's own build/test traps
    (`--no-default-features --features rocm`, never a package-wide `cargo test -p
    cb-train`) are unchanged and must be preserved by any new work.
  - The prior phase (`gpu-device-coverage-expansion`, GDC-01..12, merged into this
    branch) closed exactly two gaps: non-uniform per-object weights (all four grow
    policies) and single-permutation Borders-only CTR (two-permutation parity). It
    explicitly dropped Ordered boosting (D5) after finding the premise wrong. This
    research independently re-verified both the "still closed" state (weighted-der +
    Borders CTR) and the "still open" state of everything else directly against current
    `HEAD` source, not from memory of the prior phase's documents.
- **Highest-risk findings.**
  1. **Exact-leaf (MAE/Quantile/MAPE) is structurally dead code on the device today**,
     exactly like the five Phase-13 families — `device_config.exact_leaf` is never set to
     `true` anywhere in `crates/cb-train/src/boosting.rs` (verified: zero non-comment
     hits), and `device_host_eligible`'s final clause
     (`matches!(params.leaf_method, LeafMethod::Gradient | LeafMethod::Simple)`,
     `boosting.rs:4318`) excludes `LeafMethod::Exact` unconditionally before the
     session-side `map_leaf_method`/`DeviceLeafMethod::Exact` machinery (fully built,
     self-oracled, `crates/cb-backend/src/gpu_runtime/session.rs:980-1017`,
     `kernels/exact_quantile.rs`) is ever reached. Upstream's default leaf method for
     MAE/Quantile/MAPE is Exact, so this gap is reachable by ordinary use of those
     losses, not an exotic combination.
  2. **All five Phase-13 families (pairwise/ranking/multiclass/ordered/langevin) still
     decline unconditionally at `GpuTrainSession::begin`**, unchanged since Phase 13 and
     confirmed by re-reading `session.rs:1073-1168` line-for-line against
     `bench/kaggle_cuda_phase13.ipynb`'s own "all five decline to `Ok(None)`→CPU, the
     per-tree grow seam is a forward dependency" statement. No commit after Phase 13
     (verified via `git log --oneline -- crates/cb-backend/src/gpu_runtime/session.rs`)
     touched any of `map_pairwise_coverage`/`map_ranking_coverage`/
     `map_multiclass_coverage`/`map_langevin_coverage` except to add the (still-declining)
     Ordered arm's comment updates from GDC-14 (never executed — Track O was dropped).
  3. **Combination/tensor CTR projections are closer to reachable than they look.** The
     device accumulation kernel (`combine_projection_bins`,
     `crates/cb-backend/src/kernels/ctr_device.rs:89-117`) already folds multi-column
     projections and is exercised by the self-oracle
     `tensor_combination_ctr_matches_cpu_reference`. The blocker is entirely on the
     `cb-train` side: `ctr_types_are_device_covered` (`boosting.rs:2241-2249`) requires
     `col.projection.is_simple()`, and `build_device_ctr_config`'s column builder
     (`boosting.rs:2292-2343`) only ever extracts a SINGLE projection member
     (`vec![member]`) into `DeviceCtrColumn.member_bins`. This is "wire an existing
     kernel with new cb-train-side glue," not new device kernel design — a materially
     smaller task than Buckets/BTMV/Counter, which need genuinely new device
     accumulation semantics.
  4. **The Kaggle-verified "beats official CatBoost GPU" claim is real but fragile and
     narrow**, not comprehensive: three quoted `--features cuda` Python-API runs on the
     SAME 300k×50/depth-6/RMSE+Logloss/no-CTR/no-weights/SymmetricTree shape
     (`bench/quick_gpu_speed/kaggle-output-260716-r4{a,b,c}/report.md`) show
     `official_over_rs` speedup ratios of **1.025×/1.141×, 0.988×/0.942×, 1.050×/1.051×**
     — i.e. run-to-run noise straddles parity, and in run r4b catboost_rs was
     **slower**. No run exists at all with CTR, class weights, non-SymmetricTree grow, or
     a second loss family. A rigorous "beats upstream" claim needs a real grid across
     these axes, not a repeat of the same single shape.

## Phase Requirements

### In Scope
- Part 1: a complete gap table (below) covering every clause in `device_host_eligible`
  (`crates/cb-train/src/boosting.rs`, current span ~4214-4318) and every decline branch
  in `GpuTrainSession::begin` (`crates/cb-backend/src/gpu_runtime/session.rs`, current
  span ~1034-1450+), mapped to the corresponding CatBoost parameter(s), with kernel-
  existence classification and file:line evidence.
- Part 2: methodology requirements for a comprehensive Kaggle P100 speed-supremacy
  benchmark grid, informed by the established `bench/` harness conventions.
- Identification of the `task_type` Python-facing gap and the compile-time-vs-runtime
  backend-selection tension it creates.

### Acceptance Criteria
- No acceptance criteria are locked by any project spec for "full parameter parity" —
  this is exploratory/scoping research feeding a future spec+plan, not an existing
  in-flight spec. The working bar for any future implementation, per project-wide
  convention, is: ε=1e-4 device-vs-CPU self-oracle (D-07) for new device kernels, ≤1e-5
  vs installed upstream `catboost==1.2.10` for any new end-to-end fixture oracle.

### Out of Scope
- Implementing any of the gaps below — this document is research only.
- wgpu backend coverage — structurally excluded already (no f64/u64-atomic support); not
  re-litigated here per the task's own instruction.
- `catboost-master/` as a reference — per top-level CLAUDE.md, a stale 3-file stub, never
  a parity source. All CPU-reference claims below are grounded in this repo's own
  `crates/cb-train`/`crates/cb-compute` source or the installed `catboost==1.2.10`
  package via existing fixtures.

### Open or Conflicting Requirements
- The task prompt frames `task_type` as something that "may need to be part of scope" if
  absent. Research confirms it is fully absent (`KnownNotYet`) AND that adding it in the
  upstream sense (a per-fit runtime GPU/CPU switch) conflicts with this project's own
  locked constraint ("Backend selection: Cargo features only... no runtime switching",
  top-level CLAUDE.md). This is a genuine open decision, not a straightforward "go
  implement it" — flagged under Open Questions.
- The five Phase-13 "forward dependency" families were explicitly scoped, in Phase 13's
  own design, to land the kernel/gate first and the grow-seam wiring "later." This
  research finds no evidence a "later" phase for pairwise/ranking/multiclass/langevin was
  ever scheduled or attempted — the planner should treat this as a large backlog of
  roughly-equal-shaped work, not four small follow-ups.

## Project Constraints

- CLAUDE.md: `unwrap()` prohibited in production; `thiserror`+`anyhow`; Rust
  Builder-pattern API; Python surface both sklearn-compatible and CatBoost-native;
  **backend selection is Cargo-feature-only, no runtime switching**; oracle tolerance
  ≤1e-5 vs upstream; source/test separation mandatory (no `#[cfg(test)] mod tests` in
  production files); CubeCL kernels must use `generics-float` and consult the CubeCL
  manual (`/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md`) before
  any kernel work, and the error guideline on any build error.
- Repository-local decision IDs still load-bearing: D-04 (byte-unchanged CPU path for
  anything not admitted), D-05 (only O(1) per-level crossings), D-07 (ε=1e-4 device bar
  vs ≤1e-5 upstream bar), D-10-01/D-10-02 (per-fit all-or-nothing, classification lives
  at session construction).
- GPU tests only on `rocm`/`cuda` (`CLAUDE.md`); `--no-default-features --features rocm
  --test <name>` is the only safe invocation (package-wide `cargo test -p cb-train`
  under `--no-default-features --features rocm` fails to compile — ~52+ test files
  `use cb_backend::CpuBackend` at module scope, which is not compiled without `cpu`).
- No new external crate dependencies anticipated for the wiring-class gaps; CubeCL kernel
  authoring conventions apply to any new-kernel-class gap.

## Current Project Architecture

### The two-gate device/CPU decision (unchanged shape from the prior phase)

`device_host_eligible` (`crates/cb-train/src/boosting.rs:4214-4318`,
`[VERIFIED: LOCAL crates/cb-train/src/boosting.rs:4214]`) is a single boolean computed
once per `train_inner` call (the one function both `train`/`train_cat` funnel into,
`[VERIFIED: LOCAL grep "device_host_eligible|fn train_inner" crates/cb-train/src/boosting.rs]`
— only one binding exists in the whole file). When true AND `device_n_bins > 0`, it calls
`runtime.begin_device_training(...)` (`boosting.rs:4461-4487`), which for `GpuBackend`
funnels into `GpuTrainSession::begin` (`crates/cb-backend/src/gpu_runtime/session.rs:1034`),
the finer-grained backend coverage gate.

Full current clause list (verbatim structure, `[VERIFIED: LOCAL
crates/cb-train/src/boosting.rs:4214-4318]`):

```
let device_host_eligible = group_spans.is_none()                              // no ranking/pairwise groups
    && ordered_learning_perm.is_none()                                        // GAP: Ordered — UNCHANGED, D5 dropped it
    && ( (materialized_ctr_features.is_empty() && structure_fold_columns.iter().all(Vec::is_empty))
         || (learning_folds_for_cycle == 1
             && one_hot_bins.is_empty()
             && ctr_types_are_device_covered(&materialized_ctr_features)) )    // CTR: Borders-only, single-perm — CLOSED (GDC)
    && !penalties_active                                                      // GAP: feature-weight penalties
    && params.monotone_constraints.is_empty()                                 // GAP: monotone_constraints
    && matches!(params.grow_policy, SymmetricTree|Depthwise|Lossguide|Region)
    && approx_dimension == 1                                                  // GAP: multiclass/multi-output (forward dep)
    && !is_multiclass
    && !is_multilabel
    && (bootstrap_type==No || (bootstrap_type in {Bayesian,Bernoulli,Mvs,Poisson} && grow_policy==SymmetricTree))  // GAP: nonsym/Region x sampling
    && params.random_strength == 0.0
    && eval_sets.is_empty()                                                   // GAP: eval_sets / overfitting detector
    && has_any_scorable_feature(&matrix)
    && one_hot_cardinalities_fit_the_device(...)
    // weights.iter().all(|&w| w == 1.0) — REMOVED (GDC-05, CLOSED)
    && bias == 0.0                                                            // GAP: boost_from_average / CR-01
    && matches!(params.leaf_method, LeafMethod::Gradient | LeafMethod::Simple); // GAP: Exact (+ Newton, deliberate)
```

`GpuTrainSession::begin` (`session.rs:1034-1450+`) then independently declines, in
source order: Ordered (unconditional, `1073-1076`), pairwise (`1091-1107`, both covered
and uncovered branches decline), ranking (`1126-1134`, same shape), multiclass
(`1148-1157`, same shape), Langevin (records but does not gate,
`1158-1168`), then the grow-policy / exact-leaf / bootstrap / CTR family-default
composition (`1169-1325`).

### CTR device coverage — what closed, what stayed open

`ctr_types_are_device_covered` (`boosting.rs:2241-2249`,
`[VERIFIED: LOCAL crates/cb-train/src/boosting.rs:2241]`):
```rust
fn ctr_types_are_device_covered(cols: &[crate::ctr::CtrFeatureColumn]) -> bool {
    !cols.is_empty()
        && cols.iter().all(|col| {
            col.projection.is_simple()
                && col.ctr_type == crate::ctr::ECtrType::Borders.as_i8()
                && col.target_border_idx == 0
                && col.prior_denom == 1.0
        })
}
```
Doc comment states plainly: "the device CTR arm implements the ordered binclf `(good +
prior) / (total + 1)` statistic over SIMPLE (single-feature) projections... Everything
else (Buckets / BinarizedTargetMeanValue / Counter, combination projections,
multi-target-border Buckets columns) declines to the byte-unchanged CPU path... the
device kernels do not implement those accumulation semantics." This is a direct,
authoritative project statement that Buckets/BTMV/Counter accumulation is genuinely
unimplemented on device, not merely gated off.

`build_device_ctr_config` (`boosting.rs:2263-2361`) builds `DeviceCtrColumn` from a
SINGLE projection member (`col.projection.cat_features().first()`,
`DeviceCtrColumn { member_bins: vec![member], ... }`, `boosting.rs:2299-2335`) — this,
not a missing kernel, is why combination/tensor projections cannot reach the device: the
device kernel already handles multi-member folding
(`combine_projection_bins`, `crates/cb-backend/src/kernels/ctr_device.rs:89-117`,
`[VERIFIED: LOCAL]`), and `session.rs::build_ctr_cindex_columns` already calls it when
`col.member_bins.len() != 1` (`session.rs:207-213`).

The two-permutation machinery (structure vs. averaging) landed fully in the prior phase:
`ctr_covered` (`session.rs:130-159`) requires BOTH `ctr.averaging` and the structure half
populated with matching shape invariants; `build_device_ctr_config` populates
`averaging: Some(DeviceCtrAveraging{...})` unconditionally for any covered CTR fit
(`boosting.rs:2349-2360`). E2e oracle: `crates/cb-train/tests/device_ctr_fit_test.rs`,
fixture `crates/cb-oracle/fixtures/ctr_device_mixed/` (a NEW mixed float+cat fixture,
confirms the prior phase's finding that every OLDER CTR fixture is cat-only and
structurally unreachable — `device_n_float = matrix.n_features()`,
`has_any_scorable_feature` requires a float or one-hot column; a pure-CTR pool still
cannot reach the device today regardless of `ctr_types_are_device_covered`, since CTR
columns are additive to — not counted by — the float/one-hot eligibility check).

`learning_folds_for_cycle == 1` (`boosting.rs:4228`, wired from
`crate::learning_fold_count`, `boosting.rs:3832-3833`) is now a REAL guard (was
hardcoded `1` before the prior phase's GDC-01): `fold_count` passed to
`begin_device_training` is the real value (`boosting.rs:4471`, comment cites GDC-01
explicitly), so a `permutation_count > 1` CTR fit is correctly declined via
`ctr_covered`'s `fold_count != 1` check rather than silently riding fold-0 columns.
Multi-permutation CTR itself remains **entirely unimplemented** on device — there is no
resident-state design for a second learning fold anywhere in `cb-backend`.

### Weighted derivatives — fully closed (verified, not re-litigated)

`device_host_eligible`'s former `weights.iter().all(|&w| w == 1.0)` clause is gone
(comment at `boosting.rs:4300-4306` cites GDC-05). All four covered grow policies
(SymmetricTree via `weighted_der1_h`/`fold_weights_resident`, Depthwise/Lossguide/Region
via the caller-side `host_weighted_der1` substitution,
`session.rs:684-700`) consume `w·der1`. New tests exist:
`crates/cb-train/tests/device_weighted_fit_test.rs`,
`device_weighted_nonsym_fit_test.rs`, fixtures `weighted_device_sym/`,
`weighted_device_nonsym/`.

### Ordered boosting — confirmed unchanged (D5 finding still holds)

`session.rs:1073-1076`:
```rust
if !boosting_type_is_plain {
    let _ordered = map_ordered_coverage(loss, config, depth, fold_count);
    return Ok(None);
}
```
Unconditional decline, result discarded, exactly as documented by the prior phase's D5.
The CPU-side reason this device driver was mis-scoped is unchanged and re-confirmed this
session: `crates/cb-train/src/tree.rs:2383-2548`
(`score_candidate_ordered`/`select_level_ordered`/`greedy_tensor_search_oblivious_ordered`,
`[VERIFIED: LOCAL]`, line numbers unchanged from the prior phase's citation) shows the
REAL behavioral difference between CPU `Ordered` and `Plain` is the STRUCTURE SEARCH: for
every candidate split at every level, `score_candidate_ordered` re-assigns leaves over the
WHOLE candidate set and sums a per-segment `l2_split_score` across
`body_tail_segments(n, fold_len_multiplier)`, each with its own
`scale_l2_reg(l2, body_sum_weight, body_finish)`. Leaf values, once the structure is fixed,
take the SAME averaging-fold path as Plain. The shipped device driver
`cb_backend::gpu_runtime::ordered::accumulate_ordered_trajectory` reproduces
`ordered_approx_delta` — confirmed (again) that this function is defined and re-exported
but never called in the training loop (`grep -n ordered_approx_delta_simple
crates/cb-train/src/{boosting,tree,lib}.rs` — declaration + re-export only). A real device
Ordered arm needs a NEW per-segment ordered SPLIT-SCORE kernel, architecturally unlike the
existing oblivious per-level histogram fill (it needs `n_segments` independent
partial-sum reductions per candidate, not one).

### Phase-13 families — confirmed unchanged, HIGH confidence

`git log --oneline -- crates/cb-backend/src/gpu_runtime/session.rs` shows the only
commits after Phase-13's landing commits (`020472c`..`13d54ba`, 2026-07) that touch this
file are the prior phase's GDC commits (`3174272`, `84b4199`, `1009482`) — all CTR/weight
related, none touching `map_pairwise_coverage`/`map_ranking_coverage`/
`map_multiclass_coverage`/`map_langevin_coverage`, and Ordered's clause is explicitly
untouched per D5. `bench/kaggle_cuda_phase13.ipynb`
(`[VERIFIED: LOCAL, cell "BENCH-02 per-family speed"]`) states directly: "All five Phase-13
families decline to `Ok(None)`→CPU at the session level (the per-tree grow seam is a
forward dependency), so there is no end-to-end device train loop to time per family this
phase." This is unchanged today.

Per-family detail, all self-oracled at ε=1e-4 in isolation but unreachable from
`cb_train::train`:
- **Pairwise** (`PairLogitPairwise`, `YetiRankPairwise` — `is_pairwise_scoring`,
  `crates/cb-compute/src/ranking_der.rs:46-51`): `map_pairwise_coverage`
  (`session.rs:373-409`) computes coverage; `begin()` (`session.rs:1091-1107`) always
  returns `Ok(None)` regardless. Landed: device histogram reuse + per-leaf SPD system
  ASSEMBLY (`launch_pairwise_assemble_system_into`). Missing: the batched Cholesky SOLVE
  wiring + the per-tree pair/group descriptor crossing the `Runtime::grow_tree_on_device`
  seam (comment names this "Plan 02, GPUT-21" — never executed).
- **Ranking** (`QueryRmse`, `QuerySoftMax`, `YetiRank`, `YetiRankPairwise` is intercepted
  by the pairwise branch first): `map_ranking_coverage` (`session.rs:460-504`); decline
  at `session.rs:1126-1134`. Landed: der driver over the Plan-03 query-grouping infra
  (`crate::gpu_runtime::ranking`), self-oracled against `cb_compute::calc_ders_for_queries`.
  Missing: per-tree query-descriptor grow seam. `QueryCrossEntropy` is independently
  gated off even at the coverage-map level (`ranking_objective_covered`) — no `Loss`
  variant exists for it yet either (a third, smaller layer of gap within ranking).
- **Multiclass/multi-output** (`MultiClass`, `MultiClassOneVsAll`, `MultiLogloss`,
  `MultiCrossEntropy`, `RmseWithUncertainty`): `map_multiclass_coverage`
  (`session.rs:532-556`); decline at `session.rs:1148-1157`. Landed: `K`-dim Newton
  block-leaf driver (`grow_multiclass_block`, coupled softmax vs diagonal separable).
  Missing: the SHARED multi-dim grow seam (`Runtime::grow_tree_on_device` carries only
  scalar `approx`/`target`). `MultiQuantile` (its own exact-quantile leaf estimator) is
  not even reached by `map_multiclass_objective` — a fourth, still-smaller layer.
- **Langevin/SGLB**: `map_langevin_coverage` (`session.rs:636-663`) computed but its
  result is bound-and-discarded (`let _langevin = ...`, `session.rs:1168`) — it does not
  even gate a decline; a covered pointwise fit simply proceeds through the ordinary
  SymmetricTree gate WITHOUT noise. This is harmless only because **there is no CPU-side
  way to reach this state at all**: `grep -rn "langevin|diffusion_temperature|Langevin"
  crates/cb-train/src/*.rs crates/cb-compute/src/*.rs crates/catboost-rs/src/*.rs` —
  zero hits. `langevin`/`diffusion_temperature`/`posterior_sampling` are `KnownNotYet` in
  `crates/catboost-rs-py/src/params.rs` VOCABULARY (lines 275-277, not in `IMPLEMENTED`).
  The device kernel (`crate::kernels::langevin::langevin_covered_loss` +
  `AddLangevinNoise`) is self-oracled in total isolation from any real fit path.

### Exact-leaf device path — NEW finding this session, unreachable

`crates/cb-backend/src/gpu_runtime/session.rs:980-1017` (`DeviceLeafMethod`,
`map_leaf_method`) is a fully-built, symmetrically-documented Pattern-A gate exactly like
the other five, self-oracled via `crates/cb-backend/src/kernels/exact_quantile.rs`
(order-statistic weighted-quantile leaf; radix-sort based). It requires
`config.exact_leaf == true` to activate. `grep -n "exact_leaf"
crates/cb-train/src/boosting.rs` shows `device_config.exact_leaf` is NEVER set (stays at
`DeviceTrainConfig::default()`'s `exact_leaf: false`,
`crates/cb-compute/src/runtime.rs:1220`) — and even if it were, the
`device_host_eligible` top-level clause `matches!(params.leaf_method,
LeafMethod::Gradient | LeafMethod::Simple)` (`boosting.rs:4318`) excludes
`LeafMethod::Exact` before any of this is reached. MAE/Quantile/MAPE's documented
upstream default IS Exact (`crates/cb-compute/src/runtime.rs` `Loss::Mae`/`Quantile`/
`Mape` doc comments: "Used by the Exact leaf-estimation method"), so this is reachable by
ordinary, non-exotic use — not a forced/synthetic combination like most of the other
gaps.

### `task_type` — confirmed inert, and structurally in tension with the project's own rule

`crates/catboost-rs-py/src/params.rs`: `task_type` is in `VOCABULARY` (line 233) but NOT
in `IMPLEMENTED` (lines 66-128) ⇒ `status_of("task_type") == KnownNotYet` ⇒ Python `fit()`
rejects it as a parity gap today (per the file's own "Honesty policy" doc comment).
`devices`/`device_config` (lines 234-235) are likewise `KnownNotYet`.

Device dispatch today is entirely a **compile-time** Cargo-feature decision:
`crates/catboost-rs/src/builder.rs:1187-1190`:
```rust
#[cfg(feature = "cpu")]
let backend = CpuBackend;
#[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
let backend = GpuBackend::default();
```
A GPU-feature wheel attempts device dispatch for EVERY fit that satisfies
`device_host_eligible`, with no Python-level knob required at all; a CPU-feature wheel
can never honor `task_type="GPU"`. This matches this repo's own CLAUDE.md constraint
("Backend selection: Cargo features only... no runtime switching"). Implementing
`task_type` in the upstream sense (a per-`.fit()` GPU/CPU runtime switch) would directly
contradict that locked constraint — flagged as an explicit open question, not something
to silently resolve either way.

## Standard Stack

| Component | Version/status | Existing/Proposed | Purpose | Notes |
|---|---|---|---|---|
| CubeCL | workspace-pinned (`cubecl-core`/`cubecl-hip`/`cubecl-cuda`/`cubecl-cpu`) | Existing | GPU kernel authoring | Any NEW kernel (Ordered split-score, Buckets/BTMV/Counter accumulation, monotone/penalty-aware scoring) must follow `generics-float` and consult the CubeCL manual first per CLAUDE.md |
| `cb-compute::Runtime` trait | in-repo | Existing, needs signature extension | The `grow_tree_on_device(approx, target, sample)` seam | The SAME seam is the blocker for pairwise/ranking/multiclass/exact-leaf-if-ever-admitted — each family needs its own extra per-tree descriptor; the prior phase already established the "10-site blast radius, not 2" pattern (verified V-5 in the prior phase's PLAN.md) for any future signature change |
| `cb_core::sum_f64` | in-repo | Existing | Ordered host-side reduction primitive (D-07/D-08) | Any new host-gathered reduction (e.g. a nonsym-grow sample-multiplier fix) must route through it |
| `fold_weights_resident`/`vector_mul_kernel` | in-repo | Existing | Elementwise multiply on device | Reusable template for a bootstrap-sample-multiplier fix on nonsym/Region, mirroring the just-shipped weighted-der pattern |
| `combine_projection_bins` | in-repo, `crates/cb-backend/src/kernels/ctr_device.rs:89-117` | Existing | Multi-column CTR projection folding | Already handles combination/tensor projections; the gap is entirely `cb-train`-side glue, not a new kernel |

No new external crate dependencies are anticipated for the wiring-class gaps. Any
new-kernel-class gap (Ordered split-score, Buckets/BTMV/Counter, monotone/penalties)
stays within the existing CubeCL/`cb-backend` stack.

## Dependency Analysis

- **Direct**: `cb-train` → `cb-backend` (via `Runtime` trait object, feature-gated) →
  `cb-compute` (plain host types: `DeviceTrainConfig`, `DeviceCtrConfig`, `DeviceGrownTree`,
  the `Runtime` trait) → `cubecl`.
- **The T-10-04 feature-unification landmine still applies**: any new field added to
  `DeviceTrainConfig`/a new per-family descriptor type (e.g. a query-grouping struct for
  ranking, a K-dim block-leaf carrier for multiclass, a bias field for CR-01) must stay a
  plain host type with no `cubecl` dependency.
- **Runtime/system dependency**: `Atomic<u64>` add support (resident histogram fill) —
  unchanged constraint; rocm/cuda only.
- **No dependency additions or removals** anticipated for any gap identified here.

## Recommended Architecture and Implementation Pattern

This research does not prescribe a single implementation pattern across all gaps — the
gaps are too heterogeneous (see the four-way classification in the Summary). What IS
prescribed:

1. **Do not attempt a "close everything" phase.** The gap list below is roughly
   15-20 independently-shaped items; several (Ordered, Buckets/BTMV/Counter, monotone,
   penalties) require genuinely new kernel design each on the scale of the prior phase's
   entire CTR-averaging-permutation work. A future planner should pick ONE or a small
   coherent subset (e.g. "finish the four Phase-13 forward-dependency wirings together,
   since they share the seam-extension shape" or "close exact-leaf alone, since its gate
   is already fully built").
2. **Exact-leaf is the cheapest real win identified this session.** The device session
   side (`map_leaf_method`, `DeviceLeafMethod::Exact`, `exact_quantile.rs` kernel) is
   fully built and self-oracled. Closing it needs: (a) admit `LeafMethod::Exact` through
   `device_host_eligible`'s leaf-method clause for the covered quantile-family losses
   (MAE/Quantile/MAPE) only, (b) set `device_config.exact_leaf = true` (+ `quantile_alpha`/
   `quantile_delta`) when that combination holds, (c) a new e2e fixture (none of the
   existing device fixtures use MAE/Quantile/MAPE) at ≤1e-5 vs upstream. No new CubeCL
   kernel required.
3. **The four Phase-13 "forward dependency" families share one shape**: extend
   `Runtime::grow_tree_on_device` to carry an `Option<PerFamilyDescriptor>` (or one enum
   covering pairwise/ranking/multiclass), following the prior phase's own
   `OrderedTreeArgs`-shaped precedent (never executed, but its typed-contract pattern in
   the prior phase's SPEC.md §4 is directly reusable as a template) — do not design four
   independent seams.
4. **Combination/tensor CTR is a `cb-train`-only follow-up**, not a kernel task: extend
   `ctr_types_are_device_covered` to accept non-simple projections and
   `build_device_ctr_config`'s column builder to populate `member_bins` with ALL of
   `col.projection.cat_features()`, not just the first. Verify the online-CTR read-before-
   increment semantics are unaffected by multi-member folding (the kernel already claims
   this via its self-oracle; re-verify at the cb-train integration boundary, not assumed).
5. **CR-01 (`boost_from_average`/nonzero bias)** is a small, well-scoped fix: thread the
   real `bias` into `begin_device_training`/`GpuTrainSession::begin` and replace the
   hardcoded `vec![0.0_f64; n]` at `session.rs:1580` with `vec![bias; n]`. This does not
   require new kernel work — the resident approx handle already flows through the exact
   same code paths regardless of its initial value. Note: upstream's own
   `boost_from_average` default is `true` for RMSE, and this project's `CatBoostBuilder`
   also defaults to `true` (`builder.rs:299`) — so today, an unmodified default-config fit
   NEVER reaches the device for RMSE. This is a materially high-value fix for "typical
   usage actually reaching the device," independent of any other gap.
6. **Non-symmetric/Region × bootstrap sampling** should reuse the just-landed
   `host_weighted_der1` caller-side-substitution pattern
   (`session.rs:684-700`, verified V-2 in the prior phase's plan): compute a host sample
   multiplier the same way `bootstrap()` already does for the SymmetricTree device path,
   and fold it into the same `der1[i] * multiplier[i]` substitution at the nonsym/Region
   call sites, rather than any new kernel.
7. **Monotone constraints and feature-weight penalties both require genuinely new device
   scoring logic** (they modify the split-CANDIDATE evaluation itself, not just leaf/hist
   contribution) — no existing device precedent exists (verified via exhaustive grep for
   "monotone"/"penalt" across `crates/cb-backend/src/` — all hits are unrelated English
   usages). Treat as new-kernel-design, not wiring.
8. **eval_sets on device** has no dedicated explanatory comment anywhere near its
   exclusion clause (`boosting.rs:4272`, unlike every other clause which has one) and the
   device per-iteration grow branch (`boosting.rs:4684-4900+`) does no eval-set-approx
   bookkeeping at all. This is MEDIUM-confidence "probably tractable, not structurally
   blocked": each device-grown tree already gets a `leaf_of`-equivalent structure read
   back host-side once per tree (for Model assembly); applying that SAME structure to a
   held-out eval matrix (host-side, small — eval sets are typically much smaller than
   learn sets) to update `eval_approx` looks mechanically similar to what the CPU path
   already does. Not verified as trivial; flag as a design spike candidate, not a locked
   finding.

## Project Impact Scope

### Must Change (IF any of these gaps is chosen for implementation)
- `crates/cb-train/src/boosting.rs` — `device_host_eligible` (whichever clause(s) are
  relaxed), `device_config` construction (new descriptor fields per family), possibly
  `ctr_types_are_device_covered`/`build_device_ctr_config` (combination CTR).
- `crates/cb-backend/src/gpu_runtime/session.rs` — `begin()`'s per-family decline
  branches (pairwise/ranking/multiclass/langevin/Ordered), `ctr_types_are_device_covered`
  companion logic, the resident-approx zero-init (CR-01).
- `crates/cb-compute/src/runtime.rs` — `Runtime::grow_tree_on_device` signature (any
  seam extension), new plain-host descriptor types.
- `crates/catboost-rs-py/src/params.rs` — ONLY if `task_type` scope is resolved
  affirmatively (adds/changes `IMPLEMENTED`/vocabulary handling); otherwise untouched.
- `crates/cb-oracle/generator/gen_fixtures.py` or a new per-scenario fixture directory —
  every gap needs a NEW e2e fixture; none of the existing fixtures cover Exact-leaf,
  monotone+device, penalties+device, or any Phase-13 family end-to-end.

### May Change
- `crates/cb-backend/src/kernels/nonsym_grow.rs` / `region_device.rs` — only if
  nonsym/Region × sampling is chosen (per the `host_weighted_der1` precedent, likely NO
  changes needed inside these files, only at the `session.rs` call sites, mirroring V-2's
  finding from the prior phase).
- `crates/catboost-rs/src/builder.rs` — only if a new user-facing knob is needed (e.g. an
  Exact-leaf-on-device opt-in is likely automatic via existing `leaf_estimation_method`).

### Verification Only
- Every existing device e2e test (`device_nonsym_fit_test.rs`, `device_region_fit_test.rs`,
  `device_weighted_fit_test.rs`, `device_weighted_nonsym_fit_test.rs`,
  `device_ctr_fit_test.rs`, `bootstrap_dev_oracle_test.rs`, `ordered_test.rs`,
  `ctr_device_test.rs`) — must stay green unmodified for any gap chosen; the D-04
  byte-unchanged-for-non-admitted-configs invariant is the primary regression risk.
- `bench/kaggle_cuda_phase13.ipynb`'s existing per-family correctness self-oracles — must
  stay passing; any grow-seam extension must not regress the isolated der/solver/block
  self-oracles that gate them.

### Explicitly Out of Scope
- wgpu backend — structurally excluded (no f64/u64-atomic), not re-litigated.
- `catboost-master/` as a reference source.
- Langevin/SGLB CPU implementation — a prerequisite for any device work there, but is a
  DIFFERENT kind of gap (CPU feature parity, not GPU porting) and should be scoped
  separately if ever pursued.

## Do Not Hand-Roll

- **CTR online accumulation**: `launch_ordered_ctr_resident` + `binarize_ctr_column_resident`
  + `combine_projection_bins` — combination/tensor CTR needs cb-train-side glue only, not
  new kernel logic.
- **Elementwise multiply on device**: `fold_weights_resident`/`vector_mul_kernel` — reuse
  for any bootstrap-sample-multiplier fix on nonsym/Region.
- **Ordered summation**: `cb_core::sum_f64` — any new host-side reduction.
- **The fixed-point histogram fill/derive/score/split pipeline** — unchanged by any of
  these gaps; only the INPUT channel(s) feeding it should ever change.
- **The exact-quantile order-statistic kernel** (`kernels/exact_quantile.rs`) — fully
  built; closing the exact-leaf gap should NOT re-derive this, only wire the two
  `cb-train`-side plumbing points identified above.

## Common Pitfalls and Risks

1. **Trigger**: assuming the five Phase-13 "forward dependency" gaps are each a small
   follow-up because their der/solver/gate work already landed. **Consequence**:
   under-scoping a plan — each needs a genuinely new per-tree seam shape (query grouping
   for ranking, pair/group descriptor + Cholesky solve for pairwise, K-dim block leaf for
   multiclass) and its OWN new fixture, since none of the existing fixtures cover these
   loss families end-to-end on device. **Prevention**: size each family independently,
   reusing only the "10-site blast radius" caution the prior phase already established
   for any `Runtime` trait signature change. **Verification**: `grep -rn
   "grow_tree_on_device" crates/ --include=*.rs` before any signature change, exactly as
   the prior phase's V-5 did.
2. **Trigger**: relaxing `device_host_eligible`'s leaf-method clause to admit
   `LeafMethod::Exact` without also setting `device_config.exact_leaf`. **Consequence**:
   the device path would silently commit MAE/Quantile/MAPE fits to the ordinary
   Gradient/`calc_average` leaf formula — WRONG numerically (Exact is a weighted-median/
   quantile order statistic, not an average) — worse than today's correct CPU fallback.
   **Prevention**: both changes must land together, gated behind the SAME
   `matches!(params.loss, Mae | Quantile{..} | Mape)` condition
   `map_leaf_method` already uses. **Verification**: an e2e fixture at ≤1e-5 vs upstream
   MAE/Quantile training, asserting `device_active == true` (the "prove the gate actually
   admitted it" pattern from the prior phase's own pitfall #3).
3. **Trigger**: treating combination/tensor CTR as "just flip
   `ctr_types_are_device_covered`'s `is_simple()` check". **Consequence**: even after that
   flip, `build_device_ctr_config` still only extracts a SINGLE projection member — the
   device would silently score a combination-CTR split using only ONE of its member
   columns' bins, producing a WRONG (not just less-good) split. **Prevention**: both the
   gate relaxation AND the column-builder extension (all of
   `col.projection.cat_features()`, not `.first()`) must land together.
   **Verification**: an e2e oracle using a real combination-projection CTR fixture
   (e.g. adapting `crates/cb-oracle/fixtures/tensor_ctr_e2e/` to a device-eligible
   float+cat mix), asserting leaf VALUES/predictions match upstream, not just structure.
4. **Trigger**: implementing `task_type` as a Python kwarg that silently no-ops or, worse,
   attempts a genuine runtime backend switch. **Consequence**: either a user-facing
   parameter that looks implemented but does nothing observable (violates this project's
   own "Honesty policy" doc comment in `params.rs`), or a violation of the locked
   "Cargo-feature-only, no runtime switching" constraint. **Prevention**: get an explicit
   `AskUserQuestion` decision on what `task_type` should mean in this project before
   writing any spec. **Verification**: N/A until the decision is made.
5. **Trigger**: shipping CR-01's bias fix without re-checking every OTHER clause that
   currently composes with `bias == 0.0` implicitly (e.g. some existing device e2e test
   may currently rely on the zero-init as an unstated assumption). **Consequence**:
   a silent behavior change for an existing passing test. **Prevention**: grep every
   existing device fixture/test for `boost_from_average` before touching the clause, and
   confirm all currently-passing device tests explicitly pin `boost_from_average=False`
   (they do, per the prior phase's own fixture convention — `weighted_device_sym/`,
   `ctr_device_mixed/`, etc. — but re-verify per-fixture rather than assuming).
6. **Trigger**: reusing `bench/quick_gpu_speed/kaggle-output-260716-r4c/report.md`'s
   1.05× number as if it were a stable, reproducible "catboost_rs beats official CatBoost
   GPU" result. **Consequence**: an overclaim — the SAME shape across three runs (r4a,
   r4b, r4c) produced 1.03×/1.14×, 0.99×/0.94×, 1.05×/1.05×, i.e. genuinely noisy around
   parity. **Prevention**: any future benchmark claim must either report a
   median-of-N-runs with variance, or explicitly caveat single-run noise, per this
   project's own "do-not-fabricate" discipline (`bench/RESULTS.md`'s own house style).
   **Verification**: re-run `bench/quick_gpu_speed/bench.py` (or its successor) at least
   3× per configuration before quoting a speedup ratio in any new report.
7. **Trigger**: assuming the `Runtime::grow_tree_on_device` seam extension for one family
   (e.g. ranking) is independent of another (e.g. pairwise), since they are separate
   `Loss` variants. **Consequence**: two overlapping trait-signature PRs colliding on the
   same 10+ call sites (the prior phase's V-5 finding — the trait default, the one real
   `GpuBackend` override, ~7 test wrappers, 3 direct test calls). **Prevention**: design
   ONE seam extension (an enum or `Option<FamilyDescriptor>`) covering every family
   planned for a given phase, not one per family, and serialize work on
   `crates/cb-compute/src/runtime.rs` accordingly.

## Testing and Verification Strategy

- **Unit tests**: any new kernel (Ordered split-score, Buckets/BTMV/Counter accumulation,
  monotone/penalty scoring) needs its own isolated self-oracle at ε=1e-4 vs a
  hand-transcribed CPU reference, following the `*_device_test.rs` sibling-file
  convention (never `#[cfg(test)] mod tests` inline).
- **Integration/contract tests**: any `device_host_eligible`/`begin()` clause relaxation
  needs a paired "admits when it should" / "still declines when it shouldn't"
  assertion pair — the prior phase's `device_active == true` anti-false-pass pattern
  (assert the RIGHT arm fired, e.g. via tree-kind routing or a `CountingGpu`-style
  wrapper counting `grow_tree_on_device` calls) is mandatory, not optional.
- **End-to-end/regression tests**: every gap needs its OWN new fixture (none of the
  existing fixtures cover Exact-leaf, monotone+device, penalties+device, or any Phase-13
  family end-to-end) — generated offline against installed `catboost==1.2.10`, frozen,
  never regenerated in CI, following `crates/cb-oracle/fixtures/<scenario>/` conventions.
- **Exact commands** (verified this session, unchanged from the prior phase):
  - `cargo check -p cb-backend --no-default-features --features rocm`
  - `cargo test -p cb-backend --no-default-features --features rocm [-- <filter>]`
  - `cargo check -p cb-train --no-default-features --features rocm --test <name>`
  - `cargo test -p cb-train --no-default-features --features rocm --test <name>`
  - `cargo test --workspace` (default features, CPU regression)
  - NEVER `--features rocm` without `--no-default-features`; NEVER a package-wide
    `cargo test -p cb-train --no-default-features --features rocm` (E0432 across ~52+
    `CpuBackend`-importing test files).
- **Speed/benchmark verification** (Part 2 — see below for the grid requirements):
  `bench/quick_gpu_speed/bench.py`-style harness (real Python `.fit()` API, warm-untimed
  fit then timed fit, `predict()` call to drain the lazy CubeCL queue before stopping the
  clock) is the right TEMPLATE, but needs a real grid, not a single shape, and multiple
  repeats per cell.

## Part 2 — Kaggle P100 speed-supremacy benchmark methodology

### Established harness conventions (verified this session)

- **`bench/generator.py`** is the D-06 single-source workload generator: ONE seeded
  `generate(n_rows, n_features, seed)` (legacy `numpy.random.RandomState` Mersenne
  Twister — version-stable bytes) feeds BOTH the committed small-n correctness fixtures
  AND the large-n speed workload, so correctness and speed configs can never drift apart.
  Today it only produces: pure `float32` gaussian features, a linear-plus-noise
  regression target OR a logistic-thresholded binary target. **No categorical columns,
  no non-uniform weights, no CTR, no ranking groups exist in this generator at all.**
  A comprehensive benchmark needs generator extensions (or a second generator) for:
  categorical columns (to exercise CTR), non-uniform `sample_weight`, and grouped
  (ranking) targets.
- **`bench/quick_gpu_speed/bench.py`** is the closest thing to a "real head-to-head via
  the public Python API" harness that exists: it builds the `--features cuda` wheel via
  `maturin build --release --no-default-features --features cuda` on Kaggle (confirmed:
  `[VERIFIED: LOCAL bench/quick_gpu_speed/bench.py:220-225]`), installs it, and times
  `catboost_rs.CatBoostRegressor`/`Classifier` against `catboost.CatBoostRegressor(
  task_type="GPU")`/`Classifier` (official package) on the SAME generated data, using a
  `timed_fit()` helper that does an UNTIMED warm fit (absorb JIT/compile), then a TIMED
  fit, then a `predict()` call on a slice to drain the lazy CubeCL queue before stopping
  the clock — this is the "warm-untimed-fit then timed-fit, queue-drain via predict"
  pattern the task description references, confirmed real and in active use (3 Kaggle
  runs exist: r4a/r4b/r4c).
  - It ALSO includes a `build_eligibility_audit()` static precondition checklist and an
    explicit, always-included caveat that device activation is NOT observable from
    Python — "a silent CPU fallback therefore cannot be 100% ruled out." Any future
    benchmark must preserve this honesty discipline, not drop it because the numbers look
    good.
  - `pip install catboost` — confirmed working in this harness (`catboost==1.2.10`
    reported in every result); official CatBoost IS installable on Kaggle for a real
    head-to-head, no blocker there.
- **`bench/phase15_cuda_oracle/oracle.py`** (the `bench/RESULTS.md`/`BENCH-03-SIGNOFF.md`
  authoritative record) is a DIFFERENT, cargo-test-level methodology: correctness gates
  block BEFORE any speed number (`sys.exit(2)` on failure), single-session (one GPU/one
  driver/one seed), warm-run/JIT-excluded/queue-drained/median-of-3, and reports an
  INFORMATIONAL (never a pass/fail gate) official-CatBoost-GPU column with explicitly
  documented divergences (Region has no upstream equivalent — `N/A`, never proxied;
  `border_count` mismatch documented; quantization-cost asymmetry documented and left
  un-adjusted). This is the right rigor TEMPLATE for a future comprehensive grid, but its
  current grid is `{depthwise, region} × {depth-1, depth-6} × {n=10k..1M}` — no loss
  variation, no CTR, no weights, no ranking.
- **`bench/bootstrap_gpu/`** — a THIRD, narrower methodology: `cargo test --release`
  harness measuring specific kernel-level speedups (e.g. "parallel Poisson draw vs
  serial stream draw: 12.1-12.3×") plus an oracle-correctness-gated device-vs-CPU e2e
  check. Useful precedent for kernel-level claims, not for a whole-fit wall-clock
  comparison against official CatBoost.

### What a rigorous, comprehensive grid needs (not built anywhere today)

Axes identified as materially affecting the device path's relative speed (each backed by
a specific `device_host_eligible`/`begin()` clause or kernel dispatch difference found in
Part 1):

1. **grow_policy** — {SymmetricTree, Depthwise, Lossguide, Region}. Confirmed
   structurally different kernels/code paths (`grow_oblivious_tree_resident` vs
   `grow_nonsym_tree` vs `grow_region_tree`); `bench/RESULTS.md`'s own numbers show
   Region and Depthwise have DIFFERENT speedup profiles (29-41× vs host-CPU, not
   identical) — a single grow_policy cannot stand in for all four.
2. **loss function** — at minimum RMSE and Logloss (both already used); ideally also one
   CTR-eligible classification scenario and, if the exact-leaf gap above is ever closed,
   one MAE/Quantile scenario (a fundamentally different leaf-value kernel).
3. **presence of non-uniform weights** — now device-eligible (prior phase), but ZERO
   speed benchmark exists with weights on. The `fold_weights_resident` extra kernel
   launch per tree is a real, currently unmeasured cost.
4. **presence of CTR/categoricals** — now device-eligible for single-permutation Borders
   CTR (prior phase), but ZERO speed benchmark exists with CTR on. The two-permutation
   materialization (`build_ctr_cindex_columns` called twice at `begin()`) is a real,
   currently unmeasured fixed per-fit cost, separate from the per-tree grow cost the
   existing benchmarks measure.
5. **dataset shape (n × features)** — the existing grid already varies `n` (10k..1M) but
   ALWAYS at 20 or 50 features; feature count changes the histogram fill's per-object
   work and CubeCL launch shape differently than row count does — not interchangeable.
6. **depth** — existing grid covers depth-1 and depth-6 only; CatBoost's own common
   default is depth-6, but production models often use depth 4-10 — at minimum confirm
   the crossover point (`bench/RESULTS.md`'s own "device wins only at large n" framing)
   does not shift adversely at other depths.
7. **iterations** — existing grid pins 20-30; a very-short-iteration fit changes the
   fixed-launch-overhead-vs-total-grow-time ratio the D-10-09 "physics not tuning" note
   already flags as the dominant factor at small scale.

A representative (not exhaustive) grid honoring the project's own "physics, not tuning"
finding (device wins only above a crossover `n`, `[PROJECT: bench/RESULTS.md D-10-09]`)
would need AT LEAST: 2 grow policies × 2 losses × {weights on/off} × {CTR on/off,
where structurally reachable} × 2-3 dataset shapes at/above the known crossover ×
2 depths — on the order of 30-50 configurations, each ideally run 3× for variance
(per pitfall #6 above), which is a meaningfully larger Kaggle time budget than any single
existing run. This sizing is this research's own estimate,
`[ASSUMED — no locked project requirement pins a grid size]`, and should be confirmed
with the user (see Open Questions) rather than treated as a hard requirement.

### Kaggle/environment facts (carried from project memory, spot-checked against `bench/`)

- `pip install catboost` works on Kaggle (confirmed live in 3 separate `quick_gpu_speed`
  runs, `catboost==1.2.10` reported).
- The local `--features cuda` wheel build-and-install-via-maturin pattern is already
  proven and reusable (`bench/quick_gpu_speed/bench.py` STEP 4-5).
- `bench/RESULTS.md`'s own D-10-09 finding ("depth-1 device>=CPU only at n>=~100k;
  smaller n is physics, not a tuning gap") must be respected by any new grid — do not
  design a benchmark cell doomed to lose by construction and then report it as a
  regression.

## Planning Guidance

- Suggested work boundaries: treat Part 1 as a MENU, not a single deliverable. Recommend
  the planner pick ONE of: (a) exact-leaf alone (cheapest, gate fully built), (b) the
  four Phase-13 forward-dependency families together (share one seam-extension shape),
  (c) CR-01 bias fix alone (very small, high practical value — unblocks default-config
  RMSE fits), or (d) combination/tensor CTR alone (cb-train-only glue). Ordered boosting,
  Buckets/BTMV/Counter, monotone constraints, and penalties should each be sized as their
  OWN future research+spec pass — none of them fit inside a "parameter parity" phase
  sized like the prior one.
- Dependencies between tasks: CR-01 (bias) has NO dependency on anything else and is
  independently mergeable. Exact-leaf has no dependency on the Phase-13 families or CTR
  work. Combination/tensor CTR depends on nothing beyond the already-shipped
  two-permutation CTR machinery. The four Phase-13 families should share ONE
  `Runtime::grow_tree_on_device` signature extension (do not extend it four times).
- Decisions the planner MUST preserve: D-10-01 (per-fit all-or-nothing), D-04
  (byte-unchanged CPU path), the "admit the gate only after the kernel is numerically
  correct" ordering discipline (never flip `device_host_eligible` before the underlying
  computation is verified — this project's own prior-phase pitfall #3 lesson, reconfirmed
  applicable to every gap in Part 1).
- Items that need a spike or explicit user decision before implementation: `task_type`
  scope (see Open Questions); whether Ordered boosting's from-scratch kernel is even
  wanted given its scale (a full per-segment split-score device kernel is comparable in
  size to the ENTIRE prior CTR-averaging-permutation deliverable); the benchmark grid
  size/scope for Part 2.

## Open Questions

1. **`task_type` scope.** Given this project's own locked "Cargo-feature-only, no
   runtime switching" constraint, what should `task_type="GPU"`/`"CPU"` mean if
   implemented at all — an informational/consistency-checked no-op, a hard error when
   inconsistent with the compiled feature, or should it remain permanently `KnownNotYet`?
   This materially changes whether "task_type" belongs in scope for a "full parameter
   parity" phase at all. **Blocks**: any spec item touching `catboost-rs-py/src/params.rs`
   for `task_type`.
2. **Is Ordered boosting in scope for the NEXT phase, or deferred again?** Its real
   device work (a new per-segment split-score kernel) is comparable in size to an entire
   prior phase's CTR-averaging-permutation deliverable, not a small follow-up. The prior
   phase explicitly deferred it once already (D5). **Blocks**: Part 1 gap-table row
   "Ordered boosting" task sizing.
3. **How large should the Part 2 benchmark grid be, and how many Kaggle sessions is the
   user willing to fund?** This research's ~30-50-configuration estimate is unverified
   against any locked budget or time constraint. **Blocks**: writing a Part-2 spec with
   a concrete grid.
4. **Should Buckets/BinarizedTargetMeanValue/Counter CTR device support, monotone
   constraints, and feature-weight penalties be treated as ONE future "new device kernel"
   phase or several independent ones?** Each requires genuinely new kernel design; there
   is no existing precedent to reuse across them (unlike the four Phase-13 families,
   which share one seam shape). **Blocks**: any attempt to batch them into a single
   phase's task list.
5. **Is the eval_sets-on-device gap real (worth a spike) or should it stay permanently
   excluded?** This research found no dedicated project comment explaining WHY it is
   excluded (unlike every other clause), and no evidence it is structurally impossible —
   but also no evidence it was ever attempted. **Blocks**: whether to include it in any
   future scope, or continue treating it as "obviously fine to leave out."

## Sources

- Project documents inspected via direct file read (TreeFinder MCP was not invoked for
  this pass — the task instructions pointed directly at the prior phase's own
  `.planning/plans/gpu-device-coverage-expansion/{SPEC.md,PLAN.md,research.md}`, which
  were read in full and cross-checked against current source rather than treated as
  still-accurate):
  - `.planning/plans/gpu-device-coverage-expansion/SPEC.md` (locked decisions D1-D5,
    scope, typed contracts)
  - `.planning/plans/gpu-device-coverage-expansion/PLAN.md` (V-1..V-12 CodeGraph
    verification findings, task/wave structure)
  - `.planning/plans/gpu-device-coverage-expansion/research.md` (prior gap analysis,
    now partly superseded by this document for the CTR/weight rows)
- `[VERIFIED: LOCAL]` direct source reads and greps (this session, current `HEAD` of the
  `worktree-gpu-string-param-oracle` branch):
  - `crates/cb-train/src/boosting.rs` (lines 4050-4900+: `device_host_eligible`, device
    config construction, device grow branch; lines 2225-2361: CTR device-coverage/config
    builder; lines 3832-3833: `learning_folds_for_cycle` binding)
  - `crates/cb-backend/src/gpu_runtime/session.rs` (lines 1-700: `ctr_covered`,
    `build_ctr_cindex_columns`, coverage-map functions for pairwise/ranking/multiclass/
    ordered/langevin; lines 980-1450+: `map_leaf_method`, `begin()` full gate chain)
  - `crates/cb-backend/src/kernels/ctr_device.rs` (`combine_projection_bins`)
  - `crates/cb-train/src/tree.rs` (lines 2383-2548: `score_candidate_ordered`,
    `select_level_ordered`, `greedy_tensor_search_oblivious_ordered`)
  - `crates/cb-compute/src/runtime.rs` (`Loss` enum full variant list, `EScoreFunction`,
    `DeviceTrainConfig` defaults including `exact_leaf: false`,
    `begin_device_training`/`grow_tree_on_device` trait method signatures)
  - `crates/catboost-rs-py/src/params.rs` (IMPLEMENTED/VOCABULARY/ALIASES/
    CPU_LEGAL_CTR_TYPES registries — `task_type`, `langevin`, `diffusion_temperature`,
    `posterior_sampling` all confirmed `KnownNotYet`)
  - `crates/catboost-rs/src/builder.rs` (lines 240-340: `CatBoostBuilder::new()` defaults
    incl. `boost_from_average: true`; lines 1183-1190: compile-time backend selection)
  - `crates/cb-compute/src/ranking_der.rs` (`is_pairwise_scoring`)
  - `bench/kaggle_cuda_phase13.ipynb` (11-cell notebook, read in full via structured
    extraction — confirms all five Phase-13 families still decline)
  - `bench/RESULTS.md`, `bench/BENCH-03-SIGNOFF.md` (single-session Phase-15 P100
    aggregate speed matrix, D-10-09 crossover note)
  - `bench/quick_gpu_speed/bench.py` (the Python-API head-to-head harness, warm/timed/
    queue-drain pattern, eligibility audit + honesty caveat)
  - `bench/quick_gpu_speed/kaggle-output-260716-r4{a,b,c}/report.md` (three real runs,
    quoted speedup numbers)
  - `bench/generator.py` (D-06 single-source workload generator, current axis coverage)
  - `bench/bootstrap_gpu/kaggle-p100-poisson-260731/report.md` (kernel-level speed
    methodology precedent)
  - `git log --oneline -- crates/cb-backend/src/gpu_runtime/session.rs` (commit history
    confirming no post-Phase-13 wiring work landed for pairwise/ranking/multiclass/
    langevin)
  - `git log --oneline --follow -- bench/kaggle_cuda_phase13.ipynb`
- CodeGraph MCP (`codegraph_explore`) — used once to enumerate `Loss` enum variants,
  `EBoostingType`, `is_pairwise_scoring`, and `CatBoostBuilder` defaults; supplemented
  with direct `grep`/`Read` for everything else in this pass since the specific
  file:line-level detail needed exceeded what a single natural-language exploration call
  usefully returns for a survey of this breadth.
- Context7 CLI — not invoked. This research is entirely internal to the repository (gap
  survey of this project's own device-dispatch code) and its own benchmark harness; no
  third-party library/API documentation lookup was needed for either part.
- WebSearch/WebFetch — not invoked; no external claim in this document depends on
  information outside this repository's own source and committed benchmark artifacts.

## Confidence Assessment

- **HIGH** (directly verified against current source, multiple independent checks):
  - Ordered boosting unconditional decline (`session.rs:1073-1076`) and the real CPU
    structural difference (`tree.rs:2383-2548`).
  - All five Phase-13 families still decline unconditionally; no post-Phase-13 commit
    wired any of their grow seams.
  - Exact-leaf device path is fully built but structurally unreachable
    (`device_config.exact_leaf` never set; `LeafMethod::Exact` excluded by the top-level
    clause).
  - CTR: Borders-only/single-permutation is CLOSED; Buckets/BTMV/Counter/combination-
    projection/multi-permutation are OPEN, per the project's own explicit doc comment.
  - `task_type` is `KnownNotYet`; backend selection is compile-time-only.
  - CR-01's resident-approx zero-init is a literal, single-line hardcode
    (`session.rs:1580`).
  - The three `quick_gpu_speed` Kaggle runs' quoted speedup numbers (straddling parity).
- **MEDIUM** (supported by strong but not fully exhaustive evidence):
  - Combination/tensor CTR being "mostly wiring" rather than new kernel work — based on
    reading `combine_projection_bins`'s call site and the column-builder's single-member
    extraction, but the online-CTR read-before-increment semantics under multi-member
    folding were not independently re-derived from upstream CatBoost source in this pass.
  - eval_sets-on-device being "probably tractable" — inferred from the absence of a
    dedicated exclusion comment and the shape of the existing per-tree structure
    read-back, not from any direct attempt or design spike.
  - Non-symmetric/Region × bootstrap sampling being a small fix reusing the
    `host_weighted_der1` pattern — structurally very similar, but not verified by
    actually attempting the change.
  - The Part-2 benchmark grid size estimate (~30-50 configurations) — a reasoned
    estimate against the identified axes, not derived from any locked project
    requirement.
- **LOW** (incomplete evidence, requires validation before treating as a planning
  constraint):
  - Whether monotone constraints and feature-weight penalties could share ANY device
    infrastructure with each other or with an eventual Ordered split-score kernel (both
    modify split-candidate scoring) — plausible but not investigated at the kernel-design
    level in this pass.
  - Whether the `bench/quick_gpu_speed` noise (1.05× down to 0.94×) is dominated by
    Kaggle P100 contention/variance vs a real, fixable performance regression in some
    runs — not diagnosed here, only observed.
