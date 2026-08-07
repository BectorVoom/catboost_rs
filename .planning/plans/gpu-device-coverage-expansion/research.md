# Phase Research: GPU Device Coverage Expansion (Ordered / CTR / Weighted)

## Research Summary

- **Goal.** Close three all-or-nothing exclusion clauses in
  `device_host_eligible` (`crates/cb-train/src/boosting.rs:4083-4176`) so a fit
  using `boosting_type=Ordered`, CTR-encoded categoricals (Borders / Buckets /
  BinarizedTargetMeanValue / Counter), or non-uniform per-object weights
  (`class_weights` / `auto_class_weights` / `scale_pos_weight`) can commit to
  the device grower instead of unconditionally falling back to CPU.
- **Recommended approach.** Treat the three gaps as **three independently
  sized waves**, not one uniform relaxation:
  1. **CTR (gap 2)** is the most tractable: the backend session
     (`GpuTrainSession::begin`, `crates/cb-backend/src/gpu_runtime/session.rs`)
     already has a working, self-oracled, single-permutation CTR gate
     (`ctr_covered`, `build_ctr_cindex_columns`) that OPENS a device session
     today when handed a populated `DeviceCtrConfig`. The missing piece is
     entirely on the `cb-train` side: relax `device_host_eligible`'s
     `materialized_ctr_features.is_empty()` / `structure_fold_columns` clauses
     for the single-permutation, CPU-legal-4-type, non-one-hot case, and
     populate `DeviceTrainConfig.ctr` from `materialized_ctr_features`. **But**
     see the critical open-architecture risk below (structure-vs-averaging
     permutation divergence) before treating this as "just wire it up."
  2. **Weighted der (gap 3)** is real numerical work, exactly as scoped in the
     background: the device histogram fill and the final leaf-stat reduce
     both consume raw `der1_h` (channel 0) and a separate `weight_h`
     (channel 1) — never their elementwise product — so `calc_average` divides
     `Σder1` by `Σweight` instead of upstream's `Σ(w·der1)/Σweight`. The fix
     is bounded and the multiply kernel already exists
     (`fold_weights_resident` / `vector_mul_kernel`) but must be threaded
     through **every** device grower (oblivious resident, non-symmetric,
     Region), not just the oblivious path, and must respect the fixed-point
     histogram's documented `|Σ| < 2^33` range precondition, which weighted
     magnitudes make more likely to matter.
  3. **Ordered boosting (gap 1)** is the *largest* gap and is **not** simply an
     eligibility-flag relaxation: `GpuTrainSession::begin` **unconditionally**
     returns `Ok(None)` for any `!boosting_type_is_plain` fit
     (`session.rs:988-991`), before even consulting `map_ordered_coverage`'s
     result. The comments and the dedicated self-test
     (`begin_declines_ordered_to_cpu`, `ordered_test.rs:250-313`) both
     document this as an intentional "forward dependency": the
     `Runtime::grow_tree_on_device` seam (`crates/cb-compute/src/runtime.rs:1376-1384`)
     carries only `(approx, target, sample)` — no learn permutation, no
     per-tree dynamic body/tail boundary — which Ordered boosting's
     per-permutation historical trajectory needs every tree. Landing gap 1
     requires extending that seam (or adding a parallel one), not just
     deleting `ordered_learning_perm.is_none()` from `boosting.rs`.
- **Most important constraints.**
  - `device_host_eligible` is a single **per-fit, all-or-nothing** decision
    (D-10-01); every new clause must compose with — not silently widen — the
    existing bootstrap / grow-policy / exact-leaf / CTR mutual-exclusion
    matrix already enforced in `GpuTrainSession::begin`.
  - The device histogram is a **fixed-point `Atomic<u64>`** accumulator
    (`REDUCE_FIXEDPOINT_SCALE_F64 = 2^30`, `kernels.rs:2318-2335`) with an
    undocumented-at-the-kernel `|Σ| < 2^33` overflow precondition — directly
    relevant to gap 3 (weight magnitudes) and worth a host-side guard.
  - Build/test commands: `cargo test -p cb-train --no-default-features
    --features rocm` is the only way to reach the real device path;
    `--features rocm` alone leaves the default `cpu` feature active and is a
    **false-negative trap**. **Verified this session**: ~52 existing
    `crates/cb-train/tests/*.rs` files `use cb_backend::CpuBackend;` at module
    scope and fail `E0432` under `--no-default-features` (confirmed by
    running `cargo check -p cb-train --no-default-features --features rocm
    --test ordered_boost_e2e_oracle_test`, reproduced below). New rocm-only
    e2e tests MUST follow the `device_nonsym_fit_test.rs` /
    `device_region_fit_test.rs` precedent (avoid `CpuBackend` entirely; gate
    the whole device-exercising module behind `#[cfg(any(feature = "rocm",
    feature = "cuda"))]`; use a hand-rolled `Runtime`-implementing
    `CpuRefRuntime` for the CPU comparison) and MUST be run via `--test
    <binary_name>` selection, never a package-wide `cargo test -p cb-train`.
- **Highest-risk findings.**
  1. **Gap 1's real blocker is the `grow_tree_on_device` seam shape, not the
     `boosting.rs` eligibility clause.** `begin()` declines every Ordered fit
     unconditionally (`session.rs:988-991`), regardless of
     `map_ordered_coverage`. Landing gap 1 is an architecture task (extend the
     seam to carry per-tree body/tail boundaries + permutation), not a
     one-line relaxation.
  2. **Gap 2's CTR device gate models only ONE permutation, but upstream/
     cb-train's own CPU path needs TWO** — `materialized_ctr_features`
     (structure fold, used for the split search) and `averaging_ctr_features`
     (a **different** permutation, used to re-assign leaf membership for leaf
     VALUES, `boosting.rs:5816-5817` `assign_leaf_over_ctr_columns`). These
     two permutations are **never equal**, even at `permutation_count=1`
     (`cat_learn_permutation` = the initial shuffle `S`; `cat_averaging_permutation`
     = a *distinct* draw `Q = S∘P_avg`, `boosting.rs:3680-3750`). The existing
     `DeviceCtrConfig`/`ctr_covered` machinery captures only the structure
     permutation and would compute BOTH split AND leaf value from the same
     structure-permutation bins — a genuine, currently-unaddressed parity gap,
     not merely an untested case.
  3. **Gap 2's `fold_count` parameter is hardcoded to `1`** at the
     `begin_device_training` call site (`boosting.rs:4297`,
     `/* fold_count = */ 1,`) regardless of `params.permutation_count`. If the
     CTR eligibility clause is relaxed without also wiring the real
     `learning_folds_for_cycle` through, a multi-permutation CTR fit could
     silently commit to the device using only fold-0's structure columns,
     which is wrong for `permutation_count > 1` (`ctr_covered`'s own
     `fold_count != 1 → decline` guard would never fire, because it is always
     handed `1`).
  4. **No end-to-end weighted-training oracle fixture exists at all** —
     neither CPU nor device. `crates/cb-oracle/fixtures/class_weights/` is
     only the raw upstream `class_weights` **array** (Balanced/SqrtBalanced),
     not a trained model. `crates/cb-oracle/generator/gen_fixtures.py` has no
     `sample_weight=`/non-uniform-weight scenario anywhere. Gap 3's e2e oracle
     needs a brand-new fixture generated from scratch.
  5. Extending gap 3's weighted-der fix to the non-symmetric (Depthwise /
     Lossguide) and Region growers is in scope only if the planner chooses to
     admit non-uniform weights for those grow policies too — `grow_nonsym_tree`
     (`nonsym_grow.rs:442-448`) and the Region grower share the same
     unweighted `calc_average(sum_f64(&der_sub), sum_f64(&w_sub), scaled_l2)`
     pattern and are reachable ONLY because `device_host_eligible`'s global
     `weights.iter().all(|&w| w == 1.0)` clause currently masks them too.

## Phase Requirements

### In Scope
- Relaxing (not deleting) the following `device_host_eligible` clauses in
  `crates/cb-train/src/boosting.rs`:
  - `ordered_learning_perm.is_none()` (line 4084)
  - `materialized_ctr_features.is_empty()` (line 4085) and the companion
    `structure_fold_columns.iter().all(Vec::is_empty)` (line 4086)
  - `weights.iter().all(|&w| w == 1.0)` (line 4164)
- Wiring the already-shipped, self-oracled device kernels/drivers
  (`cb_backend::gpu_runtime::ordered::accumulate_ordered_trajectory`,
  `cb-backend/src/kernels/ctr_device.rs`) into a REAL per-fit training run
  (not just a standalone unit self-oracle).
- New weighted-derivative channel plumbing in the device histogram fill and
  leaf-value kernels (at minimum the oblivious resident path;
  `grow_oblivious_tree_resident`, `crates/cb-backend/src/gpu_runtime/mod.rs:3981-4303`).
- New end-to-end device-vs-CPU (and ideally device-vs-upstream) oracle tests
  for each of the three gaps, mirroring `device_nonsym_fit_test.rs` /
  `device_region_fit_test.rs` / `bootstrap_dev_oracle_test.rs`.
- Composing the new clauses correctly with every OTHER existing
  `device_host_eligible` / `GpuTrainSession::begin` restriction (grow_policy,
  bootstrap_type, exact_leaf, one-hot, eval_sets, monotone constraints, etc.)
  so no previously-excluded combination becomes silently admitted.

### Acceptance Criteria
- No explicit acceptance criteria are locked in a project spec for this
  specific expansion; the phase description itself supplies the working
  bar: device fits reach ≤1e-4 (the GPU bar, `D-07`) vs the CPU reference for
  structural self-oracles, and the project-wide ≤1e-5 bar vs upstream
  CatBoost 1.2.10 for any NEW end-to-end fixture oracle (matching every other
  `*_e2e_oracle_test.rs` in the repo).
- **Open**: the requesting background material does not state whether all
  three gaps must land together or may ship as independent waves. Given
  finding #1 (Ordered) and #2 (CTR two-permutation gap) are architecturally
  much larger than #3's histogram fix, the planner should confirm with the
  user whether partial delivery (e.g., weighted-der only, or CTR-without-the-
  averaging-permutation-nuance) is acceptable, or whether all three must be
  fully upstream-parity-complete.

### Out of Scope
- Multi-fold / multi-permutation CTR (`permutation_count > 1`) on the device —
  explicitly deferred upstream in the code as "Open Q3" (`ctr_covered`'s
  `fold_count != 1` decline, `session.rs:127-144`); not part of this
  expansion unless the user asks to also unlock it.
- One-hot × CTR combinations on the device — SPEC-OH-26 explicitly forbids
  mixing (`session.rs:1202-1208`, `is_covered_regime`'s
  `one_hot_flags... || ctr.is_none()` guard); must remain rejected.
- CTR × sampling (bootstrap/MVS) and CTR × exact-leaf on the device — already
  excluded in `ctr_covered`'s caller-side ANDs (`session.rs:1186-1195`); must
  remain excluded.
- Weighted der for pairwise / ranking / multiclass device losses — those
  families are already independently declined to CPU today (forward
  dependencies of their own, `session.rs:995-1072`) regardless of weight
  uniformity; out of scope unless explicitly requested.
- Ordered boosting combined with CTR on the device (`greedy_tensor_search`'s
  CTR structure search is explicitly "mutually exclusive with the Ordered
  path", `boosting.rs:5443-5444`) — CPU already treats these as mutually
  exclusive; the device work should preserve that, not attempt a combined
  arm.

### Open or Conflicting Requirements
- The background text describes gap 1's device driver as merely needing to be
  "wired" via the existing `ordered: Option<OrderedState>` "forward
  dependency" field, implying a moderate wiring task. Repository evidence
  (see Highest-risk finding #1) shows `begin()` **never even reaches** the
  point where `OrderedState` would matter — it declines before evaluating it.
  This is a **materially larger scope** than the background implies and
  should be flagged to the user/planner explicitly before task sizing.
- The background text frames gap 2 as "wire the CTR kernels into the gate";
  repository evidence shows the backend-side gate is already functional but
  captures only ONE of the CPU path's TWO required CTR permutations
  (structure vs. averaging). This is a genuine numerical-parity gap that a
  naive "populate `DeviceTrainConfig.ctr` and relax the flag" change would
  NOT close for anything beyond the narrowest test case where the two
  permutations happen to agree (which, per `boosting.rs:3680-3750`, they
  structurally never do under `need_shuffle=true`, the normal cat path).

## Project Constraints

- **CLAUDE.md / AGENTS.md**: no `unwrap()` in production code; `thiserror` +
  `anyhow`; source/test separation is mandatory — every `#[test]` must live
  in a dedicated file (`_test.rs` / `tests/*.rs`), never `#[cfg(test)] mod
  tests` at the bottom of a production file. CubeCL kernels must use
  `generics-float`, and any CubeCL kernel work must consult the CubeCL manual
  at `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md`
  first (`crates/cb-backend` kernel files already follow this).
- **Repository-local `D-*` / `WR-*` decision IDs are load-bearing** and must
  be preserved: D-04 (byte-unchanged CPU path when a config is not covered),
  D-05 (only O(1) per-level decisions cross the device/host seam; bulk data
  stays resident), D-07 (ε=1e-4 GPU bar for device-vs-CPU self-oracles vs the
  ≤1e-5 upstream bar for real oracles), D-10-01 (GPUT-01 per-fit all-or-
  nothing decision), D-10-02 (`begin()` classification lives at session
  construction), WR-01/WR-02/WR-04 (device bootstrap sampling design
  decisions and the split-scoring-vs-leaf-estimation channel separation that
  gap 3 must replicate for weights).
- **No `unwrap`/`expect`/`panic`/raw indexing** anywhere in
  `crates/cb-backend/src/**` production code (workspace lints + `D-13`); every
  device kernel launch site already follows this — new code must match.
- **Testing rule**: GPU tests only on `rocm` (per top-level CLAUDE.md); every
  precedent device e2e test in this codebase SKIPS (prints and returns,
  never panics or `#[ignore]`s) when neither `rocm` nor `cuda` is active —
  this is the repository's own "WR-01 anti-false-pass convention" and must be
  followed for any new tests.
- Dependencies: always use latest crate versions (workspace-wide constraint,
  not directly relevant to this phase since no new external deps are
  anticipated — see Dependency Analysis).

## Current Project Architecture

### The per-fit device/CPU decision (`crates/cb-train/src/boosting.rs`)

`device_host_eligible` is a single boolean computed once per `train_inner`
call, at `crates/cb-train/src/boosting.rs:4083-4176` (verified current line
numbers this session — the background's `~line 4083` estimate is exact).
Full current clause list, in source order:

```
let device_host_eligible = group_spans.is_none()                              // no ranking/pairwise groups
    && ordered_learning_perm.is_none()                                        // GAP 1
    && materialized_ctr_features.is_empty()                                   // GAP 2 (part 1)
    && structure_fold_columns.iter().all(Vec::is_empty)                       // GAP 2 (part 2)
    && !penalties_active
    && params.monotone_constraints.is_empty()
    && matches!(params.grow_policy, SymmetricTree|Depthwise|Lossguide|Region)
    && approx_dimension == 1
    && !is_multiclass
    && !is_multilabel
    && (bootstrap_type==No || (bootstrap_type in {Bayesian,Bernoulli,Mvs,Poisson} && grow_policy==SymmetricTree))
    && params.random_strength == 0.0
    && eval_sets.is_empty()
    && has_any_scorable_feature(&matrix)
    && one_hot_cardinalities_fit_the_device(...)
    && weights.iter().all(|&w| w == 1.0)                                      // GAP 3
    && bias == 0.0
    && matches!(params.leaf_method, LeafMethod::Gradient | LeafMethod::Simple);
```

`ordered_learning_perm`, `materialized_ctr_features`, and
`structure_fold_columns` are all computed earlier in the same function
(`boosting.rs:3662-3920`); see below.

When `device_host_eligible` is true AND `device_n_bins > 0`, the code calls
`runtime.begin_device_training(...)` (`boosting.rs:4292-4313`), passing a
**hardcoded** `fold_count = 1` (`boosting.rs:4297`, `/* fold_count = */ 1,`
regardless of `params.permutation_count`), the `boosting_type == Plain`
boolean, and a `DeviceTrainConfig` built at `boosting.rs:4226-4291`. This
`device_config` currently NEVER sets `.ctr` (stays `None` via
`..DeviceTrainConfig::default()`), matching the fact that the CTR
eligibility clause always excludes it upstream of this point.

Per-iteration, the device grow branch is at `boosting.rs:4509` onward (the
`for iter in resume_from..params.iterations` loop, "GPUT-01 DEVICE GROW
BRANCH", `boosting.rs:4510-4607`); when `device_active` it calls into the
runtime's per-tree seam and `continue`s, skipping the entire CPU body.

### CPU-side CTR two-permutation materialization (relevant to gap 2)

`crates/cb-train/src/boosting.rs:3662-3920`:
- `ctr_candidates = tensor_ctr_candidates(...)` (`boosting.rs:3662`).
- `cat_learn_permutation` (STRUCTURE fold-0 order) vs `cat_averaging_permutation`
  (LEAF-VALUE order `Q = S∘P_avg`) are two **distinct** permutations
  (`boosting.rs:3721-3750`), even at `permutation_count=1`.
- `structure_fold_columns: Vec<Vec<CtrFeatureColumn>>` (`boosting.rs:3837-3890`)
  is one column-set PER learning fold (length `learning_folds_for_cycle`);
  `materialized_ctr_features` (`boosting.rs:3894-3897`) aliases
  `structure_fold_columns[0]`.
- `averaging_ctr_features` (`boosting.rs:3906-3920`) is materialized under
  `cat_averaging_permutation` — a SEPARATE CTR value set used ONLY for
  leaf-membership re-assignment at leaf-value time
  (`assign_leaf_over_ctr_columns`, `boosting.rs:5816-5817`), never for the
  structure search.
- Per-iteration, `has_ctr = !materialized_ctr_features.is_empty()`
  (`boosting.rs:5447`) gates a CTR-aware structure search dispatch; the taken
  learning fold's columns (`structure_fold_columns[taken_fold]`,
  `boosting.rs:5455-5457`) drive the split search; `ctr_splits` at the end
  (`boosting.rs:6382`) records which splits were CTR-based.

This two-permutation split is THE key architecture fact for gap 2: it is not
an incidental implementation detail but a documented (`boosting.rs:3680-3710`)
requirement — the SAME split threshold is evaluated against DIFFERENT raw CTR
values depending on whether it's driving structure search or leaf-value
assignment, because the online CTR value for a given object differs by
which permutation "sees" it first (the read-before-increment CTR semantics).

### Device CTR gate + kernels (`crates/cb-backend`)

- `crates/cb-compute/src/runtime.rs:1047-1077`: `DeviceCtrColumn` (member
  bins, prior, borders) and `DeviceCtrConfig` (ONE permutation, ONE
  target_class vector, `Vec<DeviceCtrColumn>`) — both plain host types (no
  `cubecl` dependency, honoring the T-10-04 landmine note).
- `crates/cb-backend/src/gpu_runtime/session.rs:130-144` `ctr_covered`:
  requires `config.ctr.is_some()`, `permutation.len()==n`,
  `target_class.len()==n`, every column non-empty with `borders.len()+1 ==
  n_bins` (the uniform-histogram invariant).
- `crates/cb-backend/src/gpu_runtime/session.rs:157-188`
  `build_ctr_cindex_columns`: for each `DeviceCtrColumn`, folds member
  categories (`combine_projection_bins` for tensor/combination projections,
  `crates/cb-backend/src/kernels/ctr_device.rs:89-117`), launches
  `launch_ordered_ctr_resident` (the read-before-increment online CTR
  accumulation kernel) then `binarize_ctr_column_resident`, reads back ONE
  `Vec<u32>` extra cindex column per CTR column. This runs INSIDE `begin()` —
  i.e. CTR columns are materialized ONCE per fit (matches the device's
  single-permutation, single-fold covered regime), not per-tree.
- `crates/cb-backend/src/gpu_runtime/session.rs:1182-1195`: the CTR arm of
  the SymmetricTree covered-regime gate (`ctr_is_covered`) — requires
  `ctr_covered(...)`, `bootstrap_arm == NoDraw`, `!exact_leaf`,
  `mvs_lambda.is_none()`, `max_leaves.is_none()`, `!sample_from_host`. When
  satisfied, `begin()` opens a REAL session (verified via
  `session_ctr_gate_covers_single_permutation` and
  `session_ctr_augments_resident_cindex`,
  `crates/cb-backend/src/gpu_runtime/session_depth_gt1_test.rs:478-620`) —
  this is NOT a forward-dependency stub like Ordered/pairwise/ranking/
  multiclass; the session augments the resident cindex with the extra
  binarized CTR columns and the histogram fill / scorer read them like any
  other feature column (`session.rs:613-618`,
  `session.n_features_effective() == n_features + <n CTR columns>`).
- Self-oracle: `crates/cb-backend/src/kernels/ctr_device_test.rs` — 5 tests:
  `ordered_ts_matches_cpu_reference`, `first_doc_in_segment_reads_prior`,
  `one_hot_ctr_matches_cpu_reference`, `tensor_combination_ctr_matches_cpu_reference`,
  `ctr_binarized_cindex_column_bit_exact` (verified this session).

### Device Ordered driver + gate (`crates/cb-backend`)

- `crates/cb-backend/src/gpu_runtime/ordered.rs`: `OrderedTree` (borrowed
  descriptor: `leaf_of`, `der`, `weights`, `permutation`, `body_finish`,
  `tail_finish`, `n_leaves`, `scaled_l2`), `ordered_approx_delta` (host
  sequential permutation scan reproducing upstream's body-seed + tail
  add-then-read running average), `accumulate_ordered_trajectory` (resident
  device accumulation across MULTIPLE trees via the existing
  `launch_apply_leaf_delta_into` kernel, ONE final read-back).
- `crates/cb-backend/src/gpu_runtime/session.rs:524-566`: `OrderedState`
  (`#[allow(dead_code)]`) and `map_ordered_coverage` — computes whether a
  given `(loss, config, depth, fold_count)` WOULD be a covered ordered
  regime, but its result is **discarded** at the only call site.
- `crates/cb-backend/src/gpu_runtime/session.rs:974-991`: the ACTUAL gate.
  ```rust
  if !boosting_type_is_plain {
      let _ordered = map_ordered_coverage(loss, config, depth, fold_count);
      return Ok(None);
  }
  ```
  This is unconditional — EVERY Ordered fit declines to CPU regardless of
  loss/depth/config, confirmed by the self-test
  `begin_declines_ordered_to_cpu` (`ordered_test.rs:250-313`), which
  explicitly asserts a "covered ordered RMSE declines to CPU pending the
  ordered grow seam."
- `crates/cb-backend/src/gpu_runtime/session.rs:827-835`: the `ordered:
  Option<OrderedState>` field on `GpuTrainSession`, documented as "the landed
  structural seam" — never actually constructed by any real fit (only in a
  unit test at `session.rs:1559`, `ordered: None`).
- Self-oracle: `crates/cb-backend/src/gpu_runtime/ordered_test.rs` — 5 tests
  (verified this session): `device_trajectory_matches_frozen_cpu_reference`,
  `body_rows_keep_delta_zero`, `resident_trajectory_persists_across_iterations`,
  `begin_declines_ordered_to_cpu`, `single_tail_row_equals_gradient_leaf_delta`.
- The CPU-side existing e2e Ordered oracle:
  `crates/cb-train/tests/ordered_boost_e2e_oracle_test.rs` trains a
  `boosting_type=Ordered` model with `cb_train::train(&CpuBackend, ...)`
  against the frozen `crates/cb-oracle/fixtures/ordered_boost_e2e/` fixture
  (`X.npy`, `y.npy`, `model.json`, `predictions.npy`) and asserts ≤1e-5 vs
  upstream over ALL 5 iterations via `cb_model::predict_raw`. **This is the
  exact parity target and fixture the new device e2e test for gap 1 should
  reuse** — swap `CpuBackend` for a device-capable runtime and assert the
  SAME `predictions.npy` bar (once the seam exists to actually reach the
  device).

### Device weighted-der / histogram path (relevant to gap 3)

- `crates/cb-compute/src/leaf.rs:83-89` `calc_average(sum_delta, count,
  scaled_l2) = sum_delta / (count + scaled_l2)` — the SAME function used by
  CPU and device. The CPU parity target multiplies `sum_delta` by weight
  BEFORE calling it: `crates/cb-compute/src/histogram.rs:26-37` `LeafStats`
  doc: *"In the unweighted path the per-object weight is folded into `der1`
  already."* The actual fold happens at
  `crates/cb-train/src/boosting.rs:5193-5205`:
  ```rust
  let weighted_der1: Vec<f64> = ...
      ders.der1.iter().enumerate().map(|(idx, &d)| {
          let i = idx % n;
          let w = weights.get(i).copied().unwrap_or(1.0);
          d * w
      }).collect();
  ```
  This IS upstream's `TBucketStats::SumWeightedDelta` semantics — CPU leaf
  values are `calc_average(Σ(w·der1), Σw, l2)`.
- Device: `crates/cb-backend/src/gpu_runtime/mod.rs:3981-4303`
  `grow_oblivious_tree_resident` — the resident oblivious grow used for
  `grow_policy=SymmetricTree`. Per level it calls
  `launch_partition_hist2_resident_into(client, score_der1_h, score_weight_h,
  ...)` (`mod.rs:4068-4089`) where `score_der1_h`/`score_weight_h` are EITHER
  the raw `der1_h`/`weight_h` (no sampling) OR the sample-folded pair
  (`fold_weights_resident`, sampling only). **Neither path ever multiplies
  `der1_h` by `weight_h`.** At the end, leaf values are read from
  `launch_partition_update_into(client, der1_h.clone(), weight_h.clone(),
  der2_rmse_h, ...)` (`mod.rs:4236-4245`) → `part_stats` stride-3
  `[Σder1, Σweight, Σ(der2·weight)]` → `calc_average(part_stats[leaf*3],
  part_stats[leaf*3+1], scaled_l2)` (`mod.rs:4257-4264`) — i.e. `Σder1 /
  (Σweight + l2)`, the UNWEIGHTED numerator described in the phase
  background. This is BOTH the split-scoring histogram AND the final
  leaf-value reduce — gap 3 must fix both call sites (or, more precisely,
  fix `der1_h`/`score_der1_h` to already be the weighted product before
  either call).
- **The elementwise-multiply kernel already exists and must be reused**:
  `crates/cb-backend/src/kernels/bootstrap_device.rs:618-640`
  `fold_weights_resident(client, weight_h, sample_h, n)` launches
  `crate::kernels::vector_mul_kernel::launch::<f64, SelectedRuntime>(...)` —
  a generic device elementwise-multiply. The same launcher (or the same
  `fold_weights_resident` helper renamed/generalized) is the correct way to
  compute `weighted_der1_h = der1_h ⊙ weight_h` once per tree, WITHOUT
  hand-rolling a new CubeCL kernel.
- `grow_nonsym_tree` (`crates/cb-backend/src/kernels/nonsym_grow.rs:442-448`)
  and the Region grower share the SAME unweighted `calc_average(sum_f64(&der_sub),
  sum_f64(&w_sub), scaled_l2)` pattern over host-gathered `der1`/`weight`
  slices — they are reachable today only because the GLOBAL
  `weights.iter().all(|&w| w == 1.0)` clause masks them too. If gap 3 is
  scoped to ALL grow policies (not just SymmetricTree), these two also need
  the `der1[i] * weight[i]` fix at their host-gather sites.
- **Fixed-point range precondition** (directly relevant — weighted der can
  have larger magnitude than raw der): `crates/cb-backend/src/kernels.rs:2318-2335`
  `REDUCE_FIXEDPOINT_SCALE_F64 = 2^30`; the `Atomic<u64>` accumulator is exact
  only while `|Σ| < 2^33 ≈ 8.6e9` per accumulated bin; beyond that the
  wrapping `u64` add silently sign-flips with **no in-kernel guard**
  (documented at `kernels.rs:2323-2333`). This exact class of risk was
  already identified and analyzed for the bootstrap-sample-multiplier case in
  `.planning/plans/device-bootstrap-parity/SPEC.md` (Finding F-D) — the same
  reasoning applies to large `class_weights`/`scale_pos_weight` ratios.

### Weight resolution entry point (CPU-side, already shipped — PARAM-03)

`crates/catboost-rs/src/builder.rs:942` `CatBoostBuilder::resolve_weights`
combines `class_weights` / `auto_class_weights` (via
`crates/cb-data/src/weights.rs::summary_class_weights` + the Balanced/
SqrtBalanced weight function, a verbatim port of upstream
`calc_class_weights.cpp`) / `scale_pos_weight` into the final per-object
`weights: &[f64]` slice that reaches `cb_train::train`. This is the CPU-side
plumbing already shipped by PARAM-03 (commit `6f783c0`); gap 3 is purely
about making the DEVICE path accept a non-uniform result of this resolution,
not about re-deriving the weights themselves.

## Standard Stack

| Component | Version | Existing/Proposed | Purpose | Notes |
|---|---|---|---|---|
| CubeCL | workspace-pinned (`cubecl` crate family, `cubecl-core`/`cubecl-hip`/`cubecl-cpu` observed in build output) | Existing | GPU kernel authoring/dispatch | No new kernel FAMILY needed for gap 3 (reuse `vector_mul_kernel`); gap 1/2 may need new kernel launches for per-tree permutation-aware ops (Ordered) — consult the CubeCL manual before authoring |
| `cb-backend` (`rocm`/`cuda`/`wgpu`/`cpu` features) | in-repo | Existing | Backend feature-gated device dispatch | `rocm = ["cubecl/hip"]`, `default = ["cpu"]` (`crates/cb-backend/Cargo.toml:38-42`) |
| `cb-compute::Runtime` trait | in-repo | Existing | The `compute_gradients` / `begin_device_training` / `grow_tree_on_device` / `end_device_training` seam | `grow_tree_on_device` signature (`crates/cb-compute/src/runtime.rs:1376-1384`) is the gap-1 bottleneck — carries only `(approx, target, sample)` |
| `cb_core::sum_f64` | in-repo | Existing | The sanctioned ordered-summation primitive (D-07/D-08) | Any new host-side weighted-der reduction (e.g. in `grow_nonsym_tree`/Region) must route through this, not a raw fold |

No new external crates are anticipated for any of the three gaps — this is
entirely existing-infrastructure wiring plus (for gap 1/3) new CubeCL kernel
code within the existing `cb-backend` kernel modules.

## Dependency Analysis

- **Direct**: `cb-train` → `cb-backend` (via the `Runtime` trait object,
  behind Cargo features) → `cb-compute` (shared plain-host types:
  `DeviceTrainConfig`, `DeviceCtrConfig`, `DeviceGrownTree`) → `cubecl`.
- **Transitive/peer constraints already enforced**: `cb-compute` types passed
  across the `cb-train`/`cb-backend` boundary must be PLAIN HOST types (no
  `cubecl` dependency) — the repeatedly-cited "T-10-04 feature-unification
  landmine." Any new fields added to `DeviceTrainConfig`/`DeviceCtrConfig`
  for gap 1/2 (e.g. a per-tree body/tail boundary carrier for Ordered) must
  respect this.
- **Runtime/system dependency**: `Atomic<u64>` add support — the resident
  partition histogram fill requires it (`device_supports_u64_atomic_add`,
  gating `launch_partition_hist2_resident_into`); cpu/wgpu backends lack it,
  which is why every device e2e test in this codebase SKIPS on those
  backends. rocm/cuda are the only backends where gap 1/2/3 e2e tests can
  actually exercise the device path (not just the gate-classification logic,
  which runs everywhere).
- **Build dependency verified this session**: `cargo check -p cb-train
  --no-default-features --features rocm --test <name>` successfully resolves
  and compiles `cubecl-hip`/`cubecl-cpp`/`cubecl-opt` etc. on this machine
  (no ROCm runtime library needed for `cargo check`), reaching real Rust
  compile errors — i.e. this environment CAN be used to verify E0432-class
  build breakage even without a physical AMD GPU, though it cannot execute
  the rocm-gated tests themselves.
- **No dependency additions or removals** are anticipated.

## Recommended Architecture and Implementation Pattern

### Gap 3 (weighted der) — the most self-contained, do this first

1. In `GpuTrainSession` (`crates/cb-backend/src/gpu_runtime/session.rs`),
   after `der1_h` is (re-)established each tree (both the init path,
   `session.rs:1763-1774`, and the resident update path,
   `mod.rs:4272`/`session.rs:1986`), compute
   `weighted_der1_h = fold_weights_resident(&client, &der1_h, &weight_h, n)?`
   (reusing the EXISTING elementwise-multiply kernel; consider renaming/
   generalizing `fold_weights_resident` since it is no longer solely a
   bootstrap-fold helper, or add a thin same-shape wrapper).
2. Feed `weighted_der1_h` — NOT raw `der1_h` — into:
   - `launch_partition_hist2_resident_into`'s `der1_h`/`score_der1_h`
     parameter for the SPLIT histogram (both root and subtraction-trick
     levels), mirroring how `score_der1_h` already differs from `der1_h`
     under sampling (`mod.rs:4068-4089`).
   - `launch_partition_update_into`'s `der1_h` parameter for the FINAL
     leaf-stat reduce (`mod.rs:4236-4245`).
3. Leave `der1_h` (the RAW, unweighted residual) untouched for
   `launch_der_binary_resident`'s NEXT-tree recomputation
   (`mod.rs:4272`) — the residual itself (`target - approx`) is unweighted;
   only its CONTRIBUTION to the histogram/leaf sums is weighted, exactly
   mirroring the CPU split between `ders.der1` (raw) and `weighted_der1`
   (weighted) at `boosting.rs:5152-5205`.
4. If scoping to ALL grow policies (not just SymmetricTree): apply the same
   `der1[i] * weight[i]` fix at the HOST gather sites in
   `grow_nonsym_tree` (`nonsym_grow.rs:442-448`, the `der_sub`/`w_sub`
   collection before `calc_average`) and the analogous Region grower
   leaf-value computation.
5. Relax `device_host_eligible`'s `weights.iter().all(|&w| w == 1.0)` clause
   (`boosting.rs:4164`) — but consider whether to scope the relaxation to
   `grow_policy == SymmetricTree` only initially (matching how bootstrap
   sampling was scoped, `boosting.rs:4118-4125`), given item 4 above.
6. Add a host-side (or kernel-internal) magnitude guard consistent with the
   `|Σ| < 2^33` fixed-point precondition — at minimum document the bound in
   terms of `n · max(weight) · max(|der1|)` the way `device-bootstrap-parity`'s
   research/SPEC already reasoned about it for the sample multiplier.
7. New fixture: generate a trained-model oracle (splits + leaf_values +
   predictions, upstream catboost 1.2.10) with an EXPLICIT non-uniform
   `sample_weight` (or `class_weights`) and RMSE/Logloss, bias-0
   (`boost_from_average=False`, matching every other device-eligible fixture
   convention), unit `random_strength`/`bootstrap_type=No`. Add a
   `gen_*_weighted_device*()` function to
   `crates/cb-oracle/generator/gen_fixtures.py` (no existing weighted
   scenario to extend — this is new).
8. New e2e test file (NOT reusing any of the ~52 `CpuBackend`-importing
   files): mirror `device_nonsym_fit_test.rs`'s structure — module gated on
   `#[cfg(any(feature = "rocm", feature = "cuda"))]`, `GpuBackend::default()`
   for the device fit, a hand-rolled `Runtime`-impl `CpuRefRuntime` (or the
   new fixture's upstream `predictions.npy` directly) for the ≤1e-4 (device
   self-oracle) / ≤1e-5 (upstream oracle) comparison.

### Gap 2 (CTR) — tractable but must resolve the two-permutation gap first

1. **Decision needed before implementation**: does the device CTR arm target
   (a) exact upstream/cb-train CPU parity (requiring BOTH structure and
   averaging permutation materializations to reach the device, with leaf
   values gathered from a DIFFERENT bin partition than splits), or (b) an
   intentionally narrower "structure-only" approximation accepted as a
   documented, non-upstream-matching mode? Option (a) is architecturally
   substantial (the device grow's `calc_average` currently always reads
   leaf partition from the SAME cindex used for splitting — there is no
   existing device mechanism to gather leaf sums from a second, differently-
   binned CTR column). Option (b) would need explicit user sign-off since it
   breaks the ≤1e-5 upstream-parity bar this project holds itself to
   everywhere else.
2. Wire `fold_count` correctly: replace the hardcoded `/* fold_count = */ 1`
   at `boosting.rs:4297` with the real `learning_folds_for_cycle` (or an
   equivalent host-side value), so `ctr_covered`'s `fold_count != 1` decline
   is not silently bypassed for `permutation_count > 1` fits. This is
   required regardless of the (a)/(b) decision above.
3. Populate `DeviceTrainConfig.ctr = Some(DeviceCtrConfig { permutation,
   target_class, columns })` from `cat_learn_permutation` +
   `absolute_projections`/`ctr_candidates` (the structure materialization
   inputs already computed at `boosting.rs:3662-3750`), restricted to the 4
   CPU-legal, non-Counter-`Full` CTR types the background specifies
   (Borders/Buckets/BinarizedTargetMeanValue/Counter under
   `counter_calc_method=SkipTest`).
4. Relax `materialized_ctr_features.is_empty()` /
   `structure_fold_columns.iter().all(Vec::is_empty)` at `boosting.rs:4085-4086`,
   guarded additionally by: `learning_folds_for_cycle == 1` (single
   permutation only, matching `ctr_covered`'s `fold_count` requirement),
   `one_hot_bins.is_empty()` (SPEC-OH-26, already structurally true whenever
   `ctr` is populated per `is_covered_regime`'s existing assertion), and
   `!penalties_active`/no monotone constraints/no eval_sets (already covered
   by existing sibling clauses).
5. New e2e device test analogous to `device_nonsym_fit_test.rs`, seeded from
   an EXISTING CTR fixture reused/adapted:
   `crates/cb-oracle/fixtures/plain_ctr/` (Borders, single cat column,
   Logloss, permutation_count=1) or `crates/cb-oracle/fixtures/ctr_btmv_simple/`
   (BinarizedTargetMeanValue, cat-only, Logloss, permutation_count=1, already
   ships a `model.cbm` + `predictions.npy`). Both already pin
   `permutation_count=1`, `boosting_type=Plain`, matching the device-covered
   regime — no new fixture generation should be needed for the STRUCTURE
   side if option (b) above is accepted; if option (a) is required, these
   fixtures may still need inspection to confirm the structure/averaging
   permutations do NOT coincidentally agree (they should not, per
   `boosting.rs:3680-3710`, but confirm on the actual fixture's `n`/seed).

### Gap 1 (Ordered) — architecture task, plan as its own sub-phase

1. Extend `Runtime::grow_tree_on_device` (or add a sibling seam method
   specifically for ordered fits) to carry the per-tree
   `(permutation, body_finish, tail_finish)` the device
   `accumulate_ordered_trajectory`/`ordered_approx_delta` driver needs. The
   permutation is fixed per fit (from `create_folds`); `body_finish`/
   `tail_finish` are the PER-TREE dynamic body/tail boundary sequence
   (`crate::fold::body_tail_boundaries`, growing across iterations) — this
   is genuinely new per-tree state the seam does not carry today.
2. Remove the unconditional `if !boosting_type_is_plain { ...; return
   Ok(None); }` early-return at `session.rs:988-991` and let
   `map_ordered_coverage`'s `Some(OrderedState)` result actually gate
   session construction (mirroring how the CTR arm's `ctr_covered` result is
   consumed, not just computed-and-discarded).
3. Wire the resident `accumulate_ordered_trajectory`-style per-tree apply
   into the per-tree grow path (`grow_one`), analogous to how the oblivious
   resident grow applies `launch_apply_leaf_delta_into` (`mod.rs:4268`) but
   keyed by the CURRENT tree's leaf assignment scoped to the historical
   trajectory semantics, not a flat identity map (the existing
   `accumulate_ordered_trajectory` self-oracle uses an IDENTITY leaf map
   specifically because it is testing the accumulation primitive in
   isolation, not a real per-tree grow — a real grow needs the actual
   `leaf_of` from that tree's structure search).
4. Relax `ordered_learning_perm.is_none()` at `boosting.rs:4084` only once
   1-3 land; until then this clause change would be a no-op (or worse, would
   let a fit reach `begin_device_training` only to have it decline anyway,
   which is harmless but pointless).
5. New e2e device test: extend/mirror `ordered_boost_e2e_oracle_test.rs`'s
   existing `ordered_boost_e2e` fixture (already frozen, already ≤1e-5
   verified on CPU) with a `GpuBackend`-driven variant following the
   `device_nonsym_fit_test.rs` pattern — this is the ONE case among the
   three gaps where an existing e2e fixture is directly reusable without
   modification once the seam exists.

### Cross-gap composition rules the planner must preserve

- Every new clause must remain compatible with `device_host_eligible`'s
  EXISTING clauses — in particular `approx_dimension == 1`, `!is_multiclass`,
  `!is_multilabel`, `bias == 0.0`, `matches!(leaf_method, Gradient|Simple)`,
  and the grow_policy/bootstrap_type cross-product already gated at
  `boosting.rs:4089-4125`. None of the three gaps should relax any of THOSE.
- Ordered × CTR: CPU already treats these as mutually exclusive
  (`boosting.rs:5443-5444`); do not let a combined relaxation of both gap-1
  and gap-2 clauses accidentally admit an Ordered+CTR fit to the device — an
  explicit `!(ordered_active && has_ctr)` (or simply never relaxing both
  clauses simultaneously without this guard) is needed if both land.
- Weighted der (gap 3) × CTR (gap 2): CPU computes CTR leaf partitioning
  independently of sample weight, so this combination is plausible to admit
  together, but the device CTR gate's `ctr_is_covered` check
  (`session.rs:1182-1195`) currently does not interact with weight at all —
  confirm the histogram fill's weighted-der channel (gap 3) correctly reads
  the CTR-augmented cindex columns' contributions too (it should, since CTR
  columns are just additional feature columns in the SAME resident cindex,
  per `session_ctr_augments_resident_cindex`).

## Project Impact Scope

### Must Change
- `crates/cb-train/src/boosting.rs` — `device_host_eligible` clauses (lines
  4084, 4085-4086, 4164), `device_config` construction (lines 4226-4291, to
  populate `.ctr`), the hardcoded `fold_count` (line 4297). **Reason**: these
  are the actual gate; every gap's relaxation lives here. **Downstream
  effect**: any device-eligible fit's `DeviceTrainConfig` shape changes;
  every existing device e2e test that asserts on `device_active`/tree-kind
  routing must be re-verified unaffected (see Verification Only below).
- `crates/cb-backend/src/gpu_runtime/session.rs` — `begin()`'s ordered
  early-return (988-991) for gap 1; nothing structural needed for gap 2 (the
  CTR arm already exists) beyond whatever new fields `DeviceCtrConfig` needs
  if the two-permutation decision (Recommended Architecture, gap 2, item 1)
  goes with option (a).
- `crates/cb-backend/src/gpu_runtime/mod.rs` — `grow_oblivious_tree_resident`
  (weighted-der channel, gap 3).
- `crates/cb-compute/src/runtime.rs` — `Runtime::grow_tree_on_device` seam
  signature (gap 1, if extended) and/or `DeviceCtrConfig`/`DeviceTrainConfig`
  new fields.
- `crates/cb-backend/src/kernels/nonsym_grow.rs` and the Region grower
  module — ONLY if gap 3 is scoped beyond SymmetricTree.
- `crates/cb-oracle/generator/gen_fixtures.py` — new weighted-training e2e
  fixture generator (gap 3; no existing scenario to extend).

### May Change
- `crates/cb-backend/src/gpu_runtime/session.rs` `OrderedState`/`map_ordered_coverage`
  — may need new fields (e.g. carrying the fixed permutation) once gap 1's
  seam extension lands; currently `#[allow(dead_code)]`.
- `crates/cb-backend/src/kernels/bootstrap_device.rs` `fold_weights_resident`
  — possible rename/generalization if reused outside its original bootstrap
  context for gap 3 (or leave as-is and call it directly; a rename is
  cosmetic, not required).
- `crates/catboost-rs/src/builder.rs` — only if the device path needs any
  NEW user-facing knob; unlikely, since `resolve_weights`/`boosting_type`/
  categorical routing already exist and are unconditionally forwarded to
  `cb_train::BoostParams`/`train`.

### Verification Only
- Every existing device e2e test (`device_nonsym_fit_test.rs`,
  `device_region_fit_test.rs`, `bootstrap_dev_oracle_test.rs`,
  `session_depth_gt1_test.rs`, `ordered_test.rs`, `ctr_device_test.rs`,
  `session_residency.rs`) — must continue passing unchanged; none of them
  should need edits, since the planned changes are additive relaxations
  layered on top of the existing all-or-nothing gate.
  `begin_declines_ordered_to_cpu` specifically WILL need its assertions
  updated once gap 1 actually lands (it currently pins the "always declines"
  behavior as correct) — flag this as an intentional, expected test-update
  site, not a regression.
- `crates/cb-train/tests/ordered_boost_e2e_oracle_test.rs` (CPU) — must stay
  green untouched; provides the parity fixture gap 1's device variant reuses.
- `crates/catboost-rs/tests/class_weights_facade_test.rs` — facade-level
  weight-resolution test; should be unaffected by device-path changes (it
  tests `resolve_weights`, not training).

### Explicitly Out of Scope
- `catboost-master/` — per top-level CLAUDE.md, this is a stale 3-file stub,
  never a parity reference for this work.
- Multi-permutation CTR, one-hot×CTR, CTR×sampling, CTR×exact-leaf on device
  — already excluded upstream of this phase and must remain excluded (see
  Phase Requirements, Out of Scope).
- Any new external crate dependency.

## Do Not Hand-Roll

- **Elementwise multiply on device**: reuse
  `crate::kernels::vector_mul_kernel` (via `fold_weights_resident` or a
  thin same-shape wrapper) for `der1_h ⊙ weight_h` in gap 3 — do not author
  a new CubeCL kernel for this.
- **Resident trajectory accumulation**: reuse
  `launch_apply_leaf_delta_into` (already used by both the oblivious
  resident grow's approx update AND `accumulate_ordered_trajectory`) for any
  new per-tree ordered-trajectory apply in gap 1.
- **The online CTR accumulation kernel**: `launch_ordered_ctr_resident` +
  `binarize_ctr_column_resident` (`ctr_device.rs`) already implement the
  read-before-increment ordered CTR — do not reimplement CTR accumulation
  logic; if a SECOND (averaging-permutation) materialization is needed for
  gap 2 option (a), call the SAME kernel a second time with the averaging
  permutation, do not fork the kernel.
- **The ordered summation primitive**: `cb_core::sum_f64` — any new
  host-side reduction (e.g. weighted-der gather in `grow_nonsym_tree`/Region)
  must route through it, per D-07/D-08.
- **The fixed-point histogram fill/derive/score/split pipeline**
  (`launch_partition_hist2_resident_into`, `launch_derive_sibling_hist_into`,
  `score_partition_over_binsums`, `launch_partition_split_packed_into`) —
  gap 3 changes ONLY the input handle fed into the fill (weighted vs. raw
  der1), never the pipeline mechanics themselves.

## Common Pitfalls and Risks

1. **Trigger**: relaxing `materialized_ctr_features.is_empty()` without also
   fixing the hardcoded `fold_count = 1` at `boosting.rs:4297`.
   **Consequence**: a `permutation_count > 1` CTR fit silently trains on the
   device using ONLY fold-0's structure columns, diverging from the correct
   multi-fold-cycled CPU semantics without any error or warning.
   **Prevention**: thread `learning_folds_for_cycle` (or equivalent) into
   the `begin_device_training` call.
   **Verification**: a new test asserting `permutation_count > 1` + CTR
   forces CPU fallback (`device_active == false`), mirroring
   `session_ctr_gate_covers_single_permutation`'s existing
   `fold_count>1` assertion but exercised from the `cb-train` entry point.

2. **Trigger**: treating gap 2 as "populate `DeviceTrainConfig.ctr` and
   relax the flag" without addressing the structure-vs-averaging permutation
   split. **Consequence**: device CTR leaf values would be computed from the
   STRUCTURE permutation's CTR bins (the only ones resident in the device
   cindex), silently diverging from both cb-train's own CPU reference AND
   upstream beyond the narrowest structural case. **Prevention**: resolve
   the Recommended Architecture gap-2 item-1 decision explicitly with the
   user before implementation; do not silently ship option (b) as if it were
   full parity. **Verification**: an e2e oracle comparing device leaf VALUES
   (not just split structure) against the `ctr_btmv_simple`/`plain_ctr`
   fixture's `predictions.npy` at the ≤1e-5 bar — a naive implementation
   will fail this even if it passes a structure-only self-oracle.

3. **Trigger**: shipping gap 1 by simply deleting the
   `ordered_learning_perm.is_none()` clause without extending
   `Runtime::grow_tree_on_device`. **Consequence**: `begin()` still
   unconditionally returns `Ok(None)` for `!boosting_type_is_plain`
   (`session.rs:988-991`), so `device_active` stays `false` and the fit
   silently continues on CPU — the change is a no-op that looks like
   progress but changes nothing observable except removing a defensive
   early gate. **Prevention**: land the seam extension (Recommended
   Architecture gap-1 items 1-3) FIRST; only then relax the `boosting.rs`
   clause. **Verification**: assert `device_active == true` (not just "the
   fit succeeds") in the new e2e test, the same way
   `device_nonsym_fit_test.rs` asserts `dev.non_symmetric_trees.len() ==
   params.iterations` and `dev.oblivious_trees.is_empty()` to prove the
   RIGHT arm fired.

4. **Trigger**: weighted der magnitudes (large `class_weights`/
   `scale_pos_weight` ratios × large `n`) exceeding the fixed-point
   histogram's `|Σ| < 2^33` precondition. **Consequence**: silent `u64`
   wraparound corrupts the histogram with NO error surfaced (documented at
   `kernels.rs:2323-2333` as a precondition, not an enforced guard).
   **Prevention**: either add a host-side pre-launch magnitude check (as
   `kernels.rs:2333` itself suggests as an open TODO) or bound the new
   e2e fixture's weight range comfortably under the limit and document the
   residual risk. **Verification**: none exists today for this precondition
   in general — the `device-bootstrap-parity` plan flagged the same risk
   (Finding F-D) for sample multipliers and it does not appear to have been
   closed with an enforced guard either; treat as a pre-existing, orthogonal
   risk this phase inherits rather than one it must solve outright, but
   document it.

5. **Trigger**: adding a new rocm-only test file to `crates/cb-train/tests/`
   that imports `cb_backend::CpuBackend` at module (top) scope, or running
   `cargo test -p cb-train --no-default-features --features rocm` without
   `--test <name>`. **Consequence**: build failure (`E0432`) across ~52
   unrelated existing test binaries, blocking the WHOLE test run even though
   only the new file matters. **Prevention**: follow the
   `device_nonsym_fit_test.rs`/`device_region_fit_test.rs` pattern
   (hand-rolled `CpuRefRuntime`, no `CpuBackend` import, device-exercising
   code gated behind `#[cfg(any(feature = "rocm", feature = "cuda"))]`) and
   always invoke via `--test <binary_name>`. **Verification**: `cargo check
   -p cb-train --no-default-features --features rocm --test <new_test>`
   must succeed in isolation (verified this session for the existing
   precedent files: `device_nonsym_fit_test` succeeds, an arbitrary
   `CpuBackend`-importing file like `ordered_boost_e2e_oracle_test` fails
   E0432 under the same flags).

6. **Trigger**: a bare `cargo test -p cb-train --features rocm` (WITHOUT
   `--no-default-features`). **Consequence**: `cpu` stays enabled (it is
   cb-train's `default` feature, `Cargo.toml:15`), `SelectedRuntime`
   silently resolves to `cubecl-cpu`, and every "device" test either SKIPs
   (harmless false negative — looks like it ran, tested nothing) or, worse,
   compares cpu-vs-cpu and reports a false pass. **Prevention**: always pair
   `--features rocm` with `--no-default-features` for real device
   verification. **Verification**: none automated in-repo; this is a
   documented operational trap the planner should call out explicitly in
   task instructions/CI notes for this phase.

7. **Trigger**: forgetting that Ordered × CTR are CPU-mutually-exclusive
   (`boosting.rs:5443-5444`) when relaxing both gap-1 and gap-2 clauses in
   the same change. **Consequence**: a fit with BOTH `boosting_type=Ordered`
   AND CTR-routed categoricals could newly become device-eligible in a
   combination no device kernel (or CPU reference) actually implements
   together. **Prevention**: add an explicit composition guard (or simply
   sequence the two waves so this combination is never exercised without a
   deliberate design decision). **Verification**: a negative test asserting
   this combination still declines to CPU (or errors clearly) after both
   gaps land.

8. **Trigger**: extending gap 3's weighted-der fix only to
   `grow_oblivious_tree_resident` while leaving `device_host_eligible`'s
   weight clause relaxed globally (not scoped to `grow_policy ==
   SymmetricTree`). **Consequence**: Depthwise/Lossguide/Region fits with
   non-uniform weights would newly reach `grow_nonsym_tree`/the Region
   grower, which still compute UNWEIGHTED `calc_average` — a silent
   wrong-answer regression worse than today's CPU fallback. **Prevention**:
   either fix all three growers together, or explicitly scope the weight
   relaxation's `&&` to `grow_policy == SymmetricTree` (mirroring the
   existing bootstrap-type scoping pattern at `boosting.rs:4118-4125`).
   **Verification**: a test asserting non-uniform weights + Depthwise (or
   Region) still declines to CPU if the fix is scoped, or produces correct
   weighted leaf values if not.

## Testing and Verification Strategy

- **Unit/self-oracle tests** (already exist, must keep passing unchanged):
  `crates/cb-backend/src/gpu_runtime/ordered_test.rs` (5 tests),
  `crates/cb-backend/src/kernels/ctr_device_test.rs` (5 tests),
  `crates/cb-backend/src/gpu_runtime/session_depth_gt1_test.rs`'s CTR/exact/
  bootstrap gate tests, `crates/cb-compute/src/leaf_test.rs`
  (`calc_average` itself).
- **New unit-level device self-oracle** (gap 3): a device-vs-CPU comparison
  of the WEIGHTED leaf values specifically (not just structure), analogous
  in style to `session_residency_matches_cpu_multi_tree_boosting`
  (`session_residency.rs:206-323`) but with a non-uniform `weight` array
  (that test already USES `weight_mod5`, a non-trivial per-object weight —
  confirm whether it already incidentally exercises/exposes the gap-3 bug
  today, since it currently only bounds leaf-value divergence with a
  generous `LEAF_BOUND`, not a tight upstream-formula assertion).
- **Integration/e2e (per gap)**: new `crates/cb-train/tests/device_*_test.rs`
  files mirroring `device_nonsym_fit_test.rs`/`device_region_fit_test.rs`:
  train on `GpuBackend::default()` and compare against (a) a hand-rolled
  `CpuRefRuntime` at ε=1e-4 (device self-consistency) AND (b) a real upstream
  fixture's `predictions.npy` at ≤1e-5 where one exists (gap 1: reuse
  `ordered_boost_e2e`; gap 2: reuse `plain_ctr`/`ctr_btmv_simple`; gap 3: new
  fixture).
- **Regression**: assert `device_active == true` (or the equivalent
  tree-kind-routing assertions `device_nonsym_fit_test.rs` uses) so a
  silently-declining-to-CPU "fix" cannot pass by accident.
- **Exact verified commands**:
  - `cargo check -p cb-train --no-default-features --features rocm --test
    <new_test_name>` — build-verify in isolation (works on any machine,
    verified this session without ROCm hardware present).
  - `cargo test -p cb-train --no-default-features --features rocm --test
    <new_test_name>` — the real in-env run, requires an actual rocm device
    (per the background, verified working on the user's local gfx1151 rig).
  - `cargo test -p cb-backend --no-default-features --features rocm` — for
    the existing `ordered_test.rs`/`ctr_device_test.rs` self-oracles
    (per the phase background, "5 tests, all currently pass" for each).
  - Do NOT run bare `cargo test -p cb-train --no-default-features --features
    rocm` (no `--test` filter) — will fail to build due to the ~52
    `CpuBackend`-importing files.
- **Security/performance/operational**: none beyond the existing
  fixed-point-overflow precondition already flagged (pitfall 4); no new
  secrets/config/deployment surface.

## Planning Guidance

- **Suggested work boundaries**: three largely-independent tracks matching
  the three gaps, but sequence them by tractability: **gap 3 (weighted der)
  first** (bounded, self-contained, existing kernel to reuse), **gap 2 (CTR)
  second** (needs an explicit user decision on the two-permutation question
  before implementation, but the backend gate already exists), **gap 1
  (Ordered) last** (the only one requiring a `Runtime` trait seam
  extension — architecturally the largest, should probably be its own
  sub-phase/spec).
- **Dependencies between tasks**:
  - Gap 2's `fold_count` fix (item 2 in its Recommended Architecture) is a
    prerequisite for safely relaxing its eligibility clause, independent of
    the two-permutation decision.
  - Gap 1's seam extension (items 1-3) must land BEFORE its `boosting.rs`
    clause relaxation is meaningful (pitfall 3).
  - Gap 3's new fixture generator should be written before or alongside its
    e2e test, since none exists to adapt.
- **Decisions the planner must preserve**: D-04 (byte-unchanged CPU path for
  every non-newly-admitted config), D-05 (only O(1) decisions cross the
  device/host boundary per level — no gap should introduce an n-length
  per-tree read-back beyond what already exists), D-10-01 (still one
  per-fit all-or-nothing decision — do not introduce PARTIAL device commit
  within a single fit).
- **Items that require a user decision before implementation**:
  1. Gap 2: option (a) full two-permutation parity vs. option (b) narrower
     structure-only device CTR (Recommended Architecture, gap 2, item 1) —
     this changes the scope/size of gap 2 substantially and affects whether
     the ≤1e-5 upstream bar is achievable at all for the device CTR arm.
  2. Whether gap 3's weighted-der fix should extend to Depthwise/Lossguide/
     Region grow policies in this phase, or be scoped to SymmetricTree only
     (pitfall 8) — affects whether `nonsym_grow.rs`/the Region grower are
     "must change" or "out of scope."
  3. Whether all three gaps must ship together or may land as independent,
     separately-reviewable waves (Phase Requirements, Acceptance Criteria).
  4. Whether gap 1's `Runtime::grow_tree_on_device` seam extension should be
     a NEW trait method (parallel seam) or a signature change to the
     existing one — a signature change touches every other backend
     implementer (`CpuBackend`'s default impl already returns `Ok(None)`
     unconditionally, so it is low-risk, but this should be confirmed as an
     explicit design choice, not an incidental side effect).

## Open Questions

1. Does the device CTR arm need to reproduce cb-train's own CPU two-
   permutation (structure vs. averaging) leaf-value semantics exactly, or is
   a narrower structure-only device approximation acceptable for this phase?
   (Blocks gap 2's implementation shape and its achievable parity bar.)
2. Should gap 3's weighted-der fix cover all three device grow policies
   (SymmetricTree/Depthwise/Lossguide/Region) or only SymmetricTree in this
   pass? (Blocks whether `nonsym_grow.rs`/Region-grower changes are in
   scope.)
3. Must all three gaps land as one deliverable, or can they be sequenced as
   independent, separately-mergeable changes? (Affects task sizing/ordering
   and whether partial delivery — e.g. gap 3 only — is an acceptable
   outcome.)
4. Is a `Runtime` trait signature change for `grow_tree_on_device`
   acceptable, or must the ordered per-tree state travel through a new,
   additive seam method instead? (Blocks gap 1's concrete API shape.)
5. What magnitude/range guarantee (if any) should the new weighted-training
   fixture's `class_weights`/`scale_pos_weight` respect, given the
   documented-but-unenforced `|Σ| < 2^33` fixed-point histogram
   precondition? (Affects fixture design and whether a host-side guard must
   be added in this phase or can be deferred as a known limitation.)

## Sources

- Project documents inspected: `/home/user/Documents/workspace/catboost_rs/CLAUDE.md`
  (top-level and worktree copy; both consulted via the system prompt),
  `.planning/plans/device-bootstrap-parity/SPEC.md` and `research.md` (WR-01,
  the closest architectural precedent for the weighted/sampled-channel
  split), `.planning/plans/one-hot-categorical-training/` (directory
  structure only, confirmed a sibling device-wiring plan exists but was not
  the primary precedent used — device-bootstrap-parity was more directly
  on-point for the channel-splitting question).
- TreeFinder MCP: not used for this phase — this repository's specification
  store is the plain-file `.planning/plans/<slug>/` corpus (confirmed by the
  `treefinder_pending` note already present in
  `.planning/plans/device-bootstrap-parity/SPEC.md`), not a TreeFinder-
  indexed document set. Read the plain files directly instead.
- CodeGraph MCP (`codegraph_explore`): primary tool for this research —
  queries covered `device_host_eligible`/`EBoostingType`/`EGrowPolicy`
  (`crates/cb-train/src/boosting.rs`), `GpuTrainSession`/`OrderedState`/
  `map_ordered_coverage`/`ctr_covered`/`build_ctr_cindex_columns`
  (`crates/cb-backend/src/gpu_runtime/session.rs`),
  `accumulate_ordered_trajectory`/`OrderedTree` (`ordered.rs`/`ordered_test.rs`),
  `DeviceCtrConfig`/`DeviceCtrColumn`/`DeviceTrainConfig`/`Runtime` trait
  (`crates/cb-compute/src/runtime.rs`), `calc_average`/`LeafStats`/
  `reduce_leaf_stats` (`crates/cb-compute/src/leaf.rs`,
  `crates/cb-compute/src/histogram.rs`), `compute_gradients`
  (`crates/cb-backend/src/cpu_runtime.rs`), `grow_oblivious_tree_resident`/
  `launch_partition_hist2_resident_into`/`fold_weights_resident`
  (`crates/cb-backend/src/gpu_runtime/mod.rs`,
  `crates/cb-backend/src/kernels/bootstrap_device.rs`).
- Local file reads / manifest inspection / command output (`[VERIFIED]`):
  - `Read` of `crates/cb-train/src/boosting.rs` lines 3660-3920, 4055-4520,
    5150-5470, 5790-5820, 6382 (exact clause list, CTR two-permutation
    materialization, weighted_der1 computation, fold_count hardcoding).
  - `Read` of `crates/cb-backend/src/gpu_runtime/session.rs` lines 130-190,
    524-991, 1624-2000+ (`ctr_covered`, `map_ordered_coverage`, the
    unconditional Ordered decline, `grow_one`'s der1/weight handle flow).
  - `Read` of `crates/cb-backend/src/gpu_runtime/mod.rs` lines 3978-4304
    (`grow_oblivious_tree_resident` full body).
  - `Read` of `crates/cb-backend/src/gpu_runtime/session_depth_gt1_test.rs`
    lines 478-620 (`session_ctr_gate_covers_single_permutation`,
    `session_ctr_augments_resident_cindex`).
  - `Read` of `crates/cb-backend/src/gpu_runtime/ordered_test.rs` lines
    245-330 (`begin_declines_ordered_to_cpu`).
  - `Read` of `crates/cb-train/tests/ordered_boost_e2e_oracle_test.rs`,
    `device_nonsym_fit_test.rs`, `device_region_fit_test.rs` (full files —
    precedent structure).
  - `Read` of `crates/cb-compute/src/runtime.rs` lines 1079-1220, 1355-1397
    (`DeviceTrainConfig`, `is_covered_regime`, `Runtime::grow_tree_on_device`
    default).
  - `Read` of `crates/cb-compute/src/leaf.rs` lines 80-163
    (`calc_average`/`gradient_leaf_delta`).
  - `Read` of `crates/cb-backend/src/kernels.rs` lines 2318-2340
    (fixed-point range precondition).
  - `Read` of `crates/catboost-rs/src/builder.rs` lines 942+
    (`resolve_weights`) and `crates/cb-data/src/weights.rs` lines 1-60
    (`summary_class_weights`).
  - `Bash: grep -n "device_host_eligible|GPUT-01"` and related greps across
    `crates/cb-train/src/boosting.rs`, `crates/cb-backend/src/gpu_runtime/session.rs`.
  - `Bash: grep -rn "cb_backend::CpuBackend" crates/cb-train/tests/*.rs`
    (52-file list, confirmed top-level vs. function-scoped import sites).
  - `Bash: cargo check -p cb-train --no-default-features --features rocm
    --test ordered_boost_e2e_oracle_test` — **VERIFIED FAILS** with `E0432
    unresolved import cb_backend::CpuBackend` (full error captured).
  - `Bash: cargo check -p cb-train --no-default-features --features rocm
    --test device_nonsym_fit_test` — **VERIFIED SUCCEEDS**.
  - `Bash: ls crates/cb-oracle/fixtures/` and `crates/cb-oracle/generator/`
    (fixture/generator inventory: `class_weights/`, `ordered_boost_e2e/`,
    `plain_ctr/`, `ctr_btmv_simple/`, `ctr_buckets_simple/`,
    `ctr_counter_simple/`, `tensor_ctr_e2e/` present; no weighted-training
    e2e generator present).
  - `Bash: grep -n "sample_weight|weight="
    crates/cb-oracle/generator/gen_fixtures.py` — **no hits** (confirmed no
    existing non-uniform-weight e2e scenario).
  - `Bash: grep -n "auto_class_weights|scale_pos_weight"
    crates/cb-oracle/generator/gen_fixtures.py` — one hit only, inside
    `gen_class_weights()`, confirmed it produces only the raw weight-array
    fixture, not a trained model.
- Context7 CLI: not invoked — no new external library/framework is relevant
  to this phase's scope (pure internal wiring across `cb-train`/`cb-backend`/
  `cb-compute`/CubeCL, all already-adopted in-repo).
- Web: not used — no external API/library research was needed; all findings
  are grounded in repository evidence and locally-executed commands.

## Confidence Assessment

- **HIGH** (directly verified by reading current source and/or running
  commands this session):
  - Exact current line numbers and full text of every `device_host_eligible`
    clause.
  - `GpuTrainSession::begin`'s unconditional Ordered decline
    (`session.rs:988-991`) and its dedicated self-test proving this is
    intentional current behavior.
  - `ctr_covered`/`build_ctr_cindex_columns`/the CTR arm of the covered-
    regime gate already opening real device sessions (verified via reading
    `session_ctr_gate_covers_single_permutation`/
    `session_ctr_augments_resident_cindex`).
  - The CPU-side weighted-der computation (`weighted_der1`,
    `boosting.rs:5193-5205`) as the exact parity target for gap 3.
  - The device histogram fill/leaf-reduce using raw `der1_h` (never
    weight-multiplied) as the exact bug for gap 3.
  - `fold_weights_resident`/`vector_mul_kernel` as an existing, reusable
    elementwise-multiply primitive.
  - The `fold_count = 1` hardcoding at `boosting.rs:4297`.
  - The CTR two-permutation split in the CPU path
    (`cat_learn_permutation`/`cat_averaging_permutation`,
    `boosting.rs:3680-3920`) and its use in leaf-value assignment
    (`assign_leaf_over_ctr_columns`, `boosting.rs:5816-5817`).
  - The build/test command trap (`--no-default-features --features rocm`
    required; ~52 files break a package-wide test run) — reproduced with
    actual `cargo check` output this session.
  - Existing fixture/generator inventory (no weighted e2e fixture exists).
- **MEDIUM** (supported by strong in-repo documentation/comments and
  consistent code patterns, but not independently re-derived from first
  principles or executed on real device hardware this session):
  - The claim that landing gap 1 requires a `Runtime` trait seam extension
    specifically for per-tree body/tail boundaries (inferred from the
    `OrderedTree`/`accumulate_ordered_trajectory` shape plus the "forward
    dependency" comments, but the exact minimal seam shape needed is a
    design decision, not something verified by running device code).
  - The `|Σ| < 2^33` fixed-point overflow risk's practical relevance to
    realistic `class_weights`/`scale_pos_weight` magnitudes (the precondition
    itself is verified in comments; whether typical weight ranges actually
    approach it depends on dataset size/weight ratio, not verified
    numerically here).
  - Whether `session_residency_matches_cpu_multi_tree_boosting`'s existing
    `weight_mod5` fixture already incidentally exposes the gap-3 bug today
    (plausible from its generous `LEAF_BOUND`, not confirmed by running the
    test, which requires rocm/cuda hardware unavailable in this research
    session).
- **LOW** (incomplete or requires explicit user/planner decision before it
  can be treated as a constraint):
  - Whether option (a) or (b) is the right target for gap 2's CTR
    two-permutation handling — this is a genuine open design question, not
    a fact to be verified further; flagged prominently as Open Question 1.
  - Whether gap 3 should extend to non-SymmetricTree grow policies in this
    phase — likewise a scope decision, not a verifiable fact.
  - Whether all three gaps must ship as one unit — no locked project
    decision found addressing this.
