---
title: GPU Full Parameter Parity — close every device-eligibility gap, then beat official CatBoost GPU on Kaggle P100
status: draft
format: markdown
spec_version: 1
updated_at: 2026-08-07T00:00:00Z
worktree: /home/user/Documents/workspace/catboost_rs/.claude/worktrees/gpu-string-param-oracle
branch: worktree-gpu-string-param-oracle
base_commit: 1535ac1bd6bd4d7f896f0afbd3f11fb63f77d170
source_requirements:
  - "User directive (verbatim, non-negotiable): \"Full parameter implementation and beating official CatBoost on a Kaggle Notebook GPU P100.\""
  - "research.md: ./research.md — the authoritative Part-1 gap survey + Part-2 benchmark methodology assessment"
  - "Prior phase (format + convention precedent): ../gpu-device-coverage-expansion/{SPEC.md,PLAN.md}"
id_prefixes:
  - "FPP-  — Part 1, Full Parameter Parity (device-eligibility gap closure)"
  - "SPD-  — Part 2, Speed (Kaggle P100 supremacy benchmark)"
---

# GPU Full Parameter Parity

## 1. Context

`research.md` (this phase's authoritative gap survey) enumerated every remaining
exclusion clause in `crates/cb-train/src/boosting.rs::device_host_eligible` and every
decline branch in `crates/cb-backend/src/gpu_runtime/session.rs::GpuTrainSession::begin`,
and classified each by whether a device kernel already exists. It found ~15–20
independently-shaped gaps in four materially different sizes, plus a fragile,
narrow-shape speed claim that does not yet support the user's "beats official CatBoost
GPU" goal.

The user's directive is explicit and non-negotiable: **"Full parameter implementation
and beating official CatBoost on a Kaggle Notebook GPU P100."** This spec therefore
scopes **everything** research.md's Part-1 gap table found. Nothing is silently dropped
to make the phase smaller. Where a gap is genuinely enormous (a from-scratch CubeCL
kernel design), it is still IN SCOPE, but planned as a **spike-then-implement track** per
this project's own established precedent (the prior phase's Track O — see its `SPEC.md`
D5 and `PLAN.md` T18 "BLOCKED — decision required"), rather than as a fabricated
line-number-precise task breakdown for a kernel whose shape is not knowable without a
dedicated design pass.

### Current state (verified against `base_commit` 1535ac1, this session)

`device_host_eligible` (`crates/cb-train/src/boosting.rs:4214-4318`) is one boolean
computed once per `train_inner`. Verbatim clause map with current line numbers:

| line | clause | status this phase |
|---|---|---|
| 4214 | `group_spans.is_none()` | GAP — ranking / pairwise (Track R, Track W) |
| 4215 | `ordered_learning_perm.is_none()` | GAP — Ordered (Track O, spike) |
| 4225-4231 | CTR: `is_empty()` OR (`folds==1` && `one_hot_bins.is_empty()` && `ctr_types_are_device_covered`) | PARTIAL — simple Borders CLOSED; combination projections (Track K) + Buckets/BTMV/Counter (Track U, spike) still open |
| 4232 | `!penalties_active` | GAP — feature-weight penalties (Track N, spike) |
| 4233 | `params.monotone_constraints.is_empty()` | GAP — monotone constraints (Track M, spike) |
| 4237-4243 | `grow_policy ∈ {SymmetricTree, Depthwise, Lossguide, Region}` | CLOSED |
| 4244-4246 | `approx_dimension == 1 && !is_multiclass && !is_multilabel` | GAP — multi-output (Track X, spike) |
| 4263-4270 | `bootstrap_type == No` OR (sampling && `grow_policy == SymmetricTree`) | GAP — non-symmetric/Region × sampling (Track P) |
| 4271 | `params.random_strength == 0.0` | OUT (perturbation; no device RNG-per-candidate path) |
| 4272 | `eval_sets.is_empty()` | GAP — eval sets (Track V, spike) |
| 4288 | `has_any_scorable_feature(&matrix)` | CLOSED (structural) |
| 4294-4299 | `one_hot_cardinalities_fit_the_device(...)` | CLOSED (structural fallback) |
| 4312 | `bias == 0.0` | GAP — `boost_from_average` (Track A) |
| 4318 | `leaf_method ∈ {Gradient, Simple}` | GAP — Exact (Track E); Newton deliberately excluded (CR-02) |

`GpuTrainSession::begin` (`session.rs:1034+`) independently declines, in source order:
Ordered unconditionally (`1073-1076`), pairwise (`1091-1107`), ranking (`1127-1134`),
multi-output (`1149-1157`), and records-but-does-not-gate Langevin (`1168`). All five are
"forward dependency: the `Runtime::grow_tree_on_device` seam carries only
`approx`/`target`" — verbatim from the source comments at `1087-1088`, `1114-1116`,
`1143-1145`.

`crates/catboost-rs-py/src/params.rs` has `task_type` in `VOCABULARY` (line 233) but not
in `IMPLEMENTED` (lines 66-128) ⇒ `status_of("task_type") == KnownNotYet` ⇒ Python
`fit()` REJECTS it today as a parity gap (`validate_params`, `params.rs:395-423`).

### Locked decisions

Locked by the acting Planner/orchestrator for this session from the user directive plus
this project's own existing constraints. Documented here to be visible and reviewable.
They are **not** re-litigated below; a decision found infeasible against verified code is
reported in `PLAN.md` §"CORRECTIONS to SPEC.md", never silently overridden (the prior
phase's V-7/V-8/V-9 precedent).

- **D1 — `task_type` is an INFORMATIONAL/VALIDATED parameter, never a runtime switch.**
  *Rationale*: this project's own top-level `CLAUDE.md` locks "Backend selection: Cargo
  features only — `cuda`, `rocm`, `wgpu`, `cpu`; no runtime switching", and
  `crates/catboost-rs/src/builder.rs:1183-1190` implements exactly that with
  `#[cfg(feature = "cpu")] let backend = CpuBackend;` /
  `#[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))] let backend =
  GpuBackend::default();`. A genuine per-`.fit()` GPU/CPU switch is categorically
  forbidden. *Behaviour*: `task_type="CPU"` is always ACCEPTED (a no-op that matches any
  compiled feature — a CPU-feature wheel runs CPU, and a GPU-feature wheel still honours
  the request because every device-ineligible config already falls back to the
  byte-unchanged CPU grower, D-04). `task_type="GPU"` is accepted ONLY when compiled with
  a device feature (`wgpu`/`cuda`/`rocm`); on a `cpu`-only wheel it raises a clear,
  actionable error naming the required build. It is never a silent no-op — this project's
  own "Honesty policy" (`params.rs:3-18`) forbids a parameter that looks implemented but
  does nothing observable. This moves `task_type` from `KnownNotYet` to `IMPLEMENTED`.
  No device-dispatch behaviour changes: `device_host_eligible` and every gate below it
  are untouched. This is pure input validation.
- **D2 — Ordered boosting IS in scope, as a spike-then-implement track.** *Rationale*: the
  user's explicit "full parameter implementation" forbids deferring it a second time (the
  prior phase's D5 already deferred it once). But `PLAN.md` V-7 of that phase established
  — and this session re-confirmed at `crates/cb-train/src/tree.rs:2383` / `:2427` / `:2498`
  — that the real device work is a NEW per-segment ordered SPLIT-SCORE kernel mirroring
  `score_candidate_ordered`, not the shipped `accumulate_ordered_trajectory` driver.
  *Structure*: (a) a design-spike task producing a short kernel-design note, sized against
  the resident histogram substrate's existing per-level fill/score/split shape, gated with
  an explicit **"if the spike finds this exceeds ~N days-equivalent of net-new kernel
  work, STOP and report back before continuing"** instruction (this project's own
  escalation-before-overrun discipline); then (b) implementation specs written as
  goal + acceptance-bar, which the spike's own output refines — never fabricated
  line-precise steps.
- **D3 — Benchmark grid: ONE comprehensive Kaggle P100 kernel session.** *Rationale*:
  this session already hit "Maximum batch GPU session count of 2 reached" on Kaggle
  account `yensen2`; sequential/concurrent kernel churn must be minimised. *Grid*:
  `{SymmetricTree, Depthwise} × {RMSE, Logloss} × {uniform, non-uniform weights} ×
  {no CTR, single-permutation Borders CTR where the shape allows float+cat} × 2 dataset
  shapes at/above the project's own documented crossover `n`` (`bench/RESULTS.md`'s
  D-10-09: "device first beats CPU at n=100000, the smallest n tested" — do not design a
  cell doomed by construction) ≈ 16–24 cells, **each run 3× for variance** (research.md
  pitfall #6: three prior single-shape runs gave 1.025×/1.141×, 0.988×/0.942×,
  1.050×/1.051×, i.e. noise straddling parity), inside ONE Kaggle notebook script that
  builds the `--features cuda` wheel once and loops the whole grid, reusing
  `bench/quick_gpu_speed/bench.py`'s proven warm-untimed-fit / timed-fit / predict-drain
  harness and its `build_eligibility_audit()` + no-silent-fallback honesty discipline.
  `bench/generator.py` (or a sibling) is extended for categorical columns and non-uniform
  weights — it has neither today (verified: only `generate()` float32 gaussian +
  `binary_target()`).
- **D4 — Buckets/BTMV/Counter CTR + monotone constraints + feature-weight penalties are
  ONE "new device kernel" track, internally sequenced as 3 INDEPENDENT spike+implement
  sub-tracks.** *Rationale*: they are planned together now per the "full" directive, but
  research.md's own LOW-confidence finding is that they do **not** obviously share device
  infrastructure, so no shared-kernel assumption may be baked into the plan. Each is
  spiked before its implementation specs are written in detail, same spike discipline as
  D2.
- **D5 — `eval_sets`-on-device is ONE spike task.** *Rationale*: research.md rates it
  MEDIUM-confidence-tractable and it is the smallest of the uncertain items. Output is a
  design note confirming or refuting tractability; implementation specs are written in
  detail only if the spike confirms it is real.
- **D6 — Langevin/SGLB is explicitly OUT OF SCOPE for this phase.** *Rationale*:
  re-verified this session — `grep -rn "langevin|diffusion_temperature|posterior_sampling"
  crates/cb-train/src crates/cb-compute/src crates/catboost-rs/src` returns **zero** hits;
  all three are `KnownNotYet` at `crates/catboost-rs-py/src/params.rs:275-277`. There is
  no `BoostParams` field and no `train_inner` code path at all. This is a **CPU feature
  gap, not a GPU-porting gap**; device kernel work is meaningless until CPU parity exists,
  and CPU parity is an entirely different kind of phase this document must not attempt to
  scope. The self-oracled `AddLangevinNoise` kernel stays where it is.
- **D7 — Execution ordering (four waves + a guard wave).**
  - **Wave 1** — the small/well-understood wins that research.md pinned exact fix points
    for: CR-01 bias (Track A), exact-leaf gate wiring (Track E), combination/tensor CTR
    glue (Track K), non-symmetric/Region × sampling (Track P), and the **single**
    `Runtime::grow_tree_on_device` signature change that serves all Phase-13 forward-
    dependency families (Track S) — research.md pitfall #7 explicitly warns against four
    separate signature changes. These are genuinely GDC-shaped and get full Red/Green/
    Refactor detail.
  - **Wave 2** — `task_type` (D1). Independently mergeable, no dependency on Wave 1.
  - **Wave 3** — the benchmark (D3). Depends on Wave 1 landing first, because the
    benchmark must measure the NEWLY reachable configs, not just what the prior phase
    already shipped.
  - **Wave 4** — the spike tracks (D2 Ordered, D4 Buckets/BTMV/Counter + monotone +
    penalties, D5 eval_sets, plus the per-family Phase-13 grow consumption for pairwise /
    ranking / multi-output). Largest, most uncertain, explicitly LAST. Each spike is gated
    to report back rather than silently ballooning scope.
  - **Wave 5** — cross-gap composition regression guards.

## 2. Scope and non-goals

### In scope

**Part 1 — Full parameter parity (`FPP-`)**

- Track A — `boost_from_average` / non-zero bias on device (CR-01).
- Track E — `leaf_estimation_method=Exact` on device for the covered quantile family.
- Track K — combination/tensor CTR projections on device (`cb-train`-side glue).
- Track P — non-symmetric / Lossguide / Region × bootstrap sampling on device.
- Track S — ONE `Runtime::grow_tree_on_device` signature extension carrying a per-family
  descriptor for every Phase-13 forward-dependency family.
- Track T — `task_type` as an informational/validated Python parameter (D1).
- Track O — Ordered boosting on device (spike → implement, D2).
- Track U — Buckets / BinarizedTargetMeanValue / Counter CTR accumulation on device
  (spike → implement, D4).
- Track M — `monotone_constraints` on device (spike → implement, D4).
- Track N — feature-weight penalties (`feature_weights`, `first_feature_use_penalties`,
  `per_object_feature_penalties`, `penalties_coefficient`) on device (spike → implement,
  D4).
- Track V — `eval_sets` + overfitting detector on device (spike, D5).
- Track W — pairwise (`PairLogitPairwise`, `YetiRankPairwise`) grow consumption (spike →
  implement).
- Track R — ranking (`QueryRMSE`, `QuerySoftMax`, `YetiRank`) grow consumption (spike →
  implement).
- Track X — multi-output (`MultiClass`, `MultiClassOneVsAll`, `MultiLogloss`,
  `MultiCrossEntropy`, `RMSEWithUncertainty`) grow consumption (spike → implement).
- Track G — cross-gap composition and regression guards.
- A NEW, offline-generated, frozen `catboost==1.2.10` fixture for **every** gap: no
  existing fixture covers Exact-leaf-on-device, monotone+device, penalties+device,
  `task_type`, non-zero-bias-on-device, combination-CTR-on-device, or any Phase-13 family
  end-to-end (research.md, re-confirmed this session).

**Part 2 — Speed supremacy (`SPD-`)**

- Extending the D-06 single-source workload generator with categorical columns and
  non-uniform sample weights.
- A grid benchmark harness (D3's 16–24 cells × 3 repeats) preserving
  `bench/quick_gpu_speed/bench.py`'s warm/timed/queue-drain pattern, its
  `build_eligibility_audit()` static precondition checklist, and its explicit
  "device activation is not observable from Python, so a silent CPU fallback cannot be
  100% ruled out" caveat.
- ONE Kaggle P100 session executing the whole grid and producing a
  median-of-3-with-variance report under `bench/`'s existing house style.

### Out of scope

- **Langevin / SGLB (`langevin`, `diffusion_temperature`, `posterior_sampling`) —
  entirely (D6).** A CPU feature gap, not a GPU-porting gap. No `BoostParams` field, no
  `train_inner` path, `KnownNotYet` in the Python registry. Device wiring is meaningless
  until a CPU parity target exists.
- `random_strength != 0.0` on device — per-candidate score perturbation has no device
  RNG-per-candidate path and is not in research.md's closable set.
- `permutation_count > 1` (multi-permutation) CTR on device — no resident-state design
  for a second learning fold exists anywhere in `cb-backend`.
- One-hot × CTR on device (SPEC-OH-26 forbids the mixed pool outright).
- Newton leaf on Logloss/CrossEntropy on device (CR-02, deliberate: numerically diverges
  from the device's `calc_average` leaf formula).
- `wgpu` backend coverage — structurally excluded (no f64 / u64-atomic support).
- `catboost-master/` as any kind of reference (stale 3-file stub, per top-level
  `CLAUDE.md`).
- A host-side enforced guard against the fixed-point histogram's `|Σ| < 2^33` overflow
  precondition — a documented residual this phase inherits (see §9), not one it closes.
- Any new external crate dependency.

## 3. Dependencies

- `crates/cb-train` → `crates/cb-backend` (via the `Runtime` trait object, behind Cargo
  features) → `crates/cb-compute` (plain-host shared types: `DeviceTrainConfig`,
  `DeviceCtrConfig`, `DeviceGrownTree`, the `Runtime` trait) → `cubecl`
  (workspace-pinned; `rocm = ["cubecl/hip"]`, `cuda = ["cubecl/cuda"]`,
  `default = ["cpu"]` on `cb-backend`).
- `crates/catboost-rs-py` → `catboost-rs` with a full `cpu`/`rocm`/`wgpu`/`cuda` feature
  passthrough (verified, `crates/catboost-rs-py/Cargo.toml` `[features]`) — so D1's
  compile-time `task_type` check is expressible directly in `params.rs` with
  `#[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]`, with **no** new
  dependency and **no** change to the backend-selection architecture.
- **T-10-04 feature-unification landmine (still binding)**: every type crossing the
  `cb-train`/`cb-backend` boundary must stay a plain host type with no `cubecl`
  dependency. This governs the new `DeviceTrainConfig.bias` field (Track A), the exact-leaf
  config fields (Track E), and the new per-family descriptor carried by the extended
  `grow_tree_on_device` seam (Track S).
- `Atomic<u64>` add support is required for the resident partition histogram fill; cpu and
  wgpu backends lack it. Every new device e2e test must SKIP (print + return, never panic,
  never `#[ignore]`) off rocm/cuda — the WR-01 anti-false-pass convention.
- No new external crate dependencies anticipated for any track, including the spike
  tracks (any new kernel stays within the existing CubeCL/`cb-backend` stack).

## 4. Typed contracts

```rust
// crates/cb-compute/src/runtime.rs — DeviceTrainConfig, EXTENDED.
//
// NOTE the project's OWN stated convention, verbatim from the `begin_device_training`
// doc comment (runtime.rs:1353-1355): "the config surface widens by mutating this
// struct, not the argument list". Both new knobs below therefore ride
// `DeviceTrainConfig` rather than growing an already-`#[allow(clippy::too_many_arguments)]`
// 13-parameter signature.
struct DeviceTrainConfig {
    // ... existing fields unchanged ...

    /// FPP-01 (CR-01). The per-fit starting approximant (`starting_approx`,
    /// `crates/cb-train/src/boosting.rs:3641`). `0.0` — the current, byte-unchanged
    /// default — reproduces today's `vec![0.0_f64; n]` resident seed exactly (D-04).
    /// A non-zero value seeds the resident approx to `vec![bias; n]` so a
    /// `boost_from_average=true` fit (upstream's RMSE default, and this project's
    /// `CatBoostBuilder` default at `builder.rs:299`) can reach the device.
    ///
    /// PLAIN HOST `f64`, no cubecl (T-10-04).
    bias: f64,                          // NEW, FPP-01. default: 0.0

    /// EXISTING but never set by cb-train today (`runtime.rs:1162`, default `false`
    /// at `:1220`). FPP-05 makes it load-bearing.
    exact_leaf: bool,
    quantile_alpha: f64,                // EXISTING, default 0.5
    quantile_delta: f64,                // EXISTING, default 1e-6
}
```

```rust
// crates/cb-compute/src/runtime.rs — Runtime trait, EXTENDED (Track S / FPP-12).
//
// ONE signature change serving EVERY Phase-13 forward-dependency family. research.md
// pitfall #7 explicitly forbids four independent seam extensions colliding on the same
// call sites.
trait Runtime {
    // ... compute_gradients, compute_gradients_grouped, begin_device_training, unchanged ...

    /// EXTENDED (FPP-12): now additionally carries the per-tree, per-family descriptor
    /// the pairwise / ranking / multi-output growers need and the scalar seam cannot
    /// express. `None` for EVERY pointwise scalar fit — the byte-unchanged D-04 default
    /// path — so every currently-green test is a pure mechanical `None` addition.
    fn grow_tree_on_device(
        &self,
        approx: &[f64],
        target: &[f64],
        sample: &[f64],
        family: Option<&FamilyTreeArgs<'_>>,   // NEW parameter, FPP-12
    ) -> CbResult<Option<DeviceGrownTree>> {
        let _ = (approx, target, sample, family);
        Ok(None)
    }
}

/// NEW plain-host enum (FPP-12), `crates/cb-compute/src/runtime.rs`.
/// One variant per Phase-13 family. Plain host types only — no `cubecl`, no `cb-train`
/// type may appear here (T-10-04). All slices are borrowed from cb-train-owned buffers
/// for the duration of the call; nothing is retained by the backend.
enum FamilyTreeArgs<'a> {
    /// Track R. Query grouping for QueryRMSE / QuerySoftMax / YetiRank.
    /// `group_offsets` is a CSR-style prefix over object order, length `n_groups + 1`.
    Ranking {
        group_offsets: &'a [u32],
    },
    /// Track W. Per-object pair/group descriptor for PairLogitPairwise / YetiRankPairwise,
    /// plus the per-tree re-sampled competitor adjacency the CPU path rebuilds from the
    /// CURRENT approx before every tree (`boosting.rs:5278-5281`).
    Pairwise {
        group_offsets: &'a [u32],
        pair_begin: &'a [u32],
        pair_end: &'a [u32],
        pair_weight: &'a [f64],
    },
    /// Track X. The `K`-dimensional approximant for MultiClass / MultiClassOneVsAll /
    /// MultiLogloss / MultiCrossEntropy / RMSEWithUncertainty. `approx_k` is DIM-MAJOR
    /// (`approx_k[d * n + i]`), matching cb-train's own multi-output approx buffer
    /// layout (`boosting.rs:3649-3660`). The returned `DeviceGrownTree.leaf_values` is
    /// already a `leaf_count × approx_dim` ROW-MAJOR block (existing contract,
    /// `runtime.rs` `DeviceGrownTree::leaf_values` doc) — no return-type change needed.
    MultiOutput {
        approx_k: &'a [f64],
        approx_dim: usize,
    },
}
```

```rust
// crates/cb-train/src/boosting.rs — device_host_eligible, RELAXED clauses.
// Current line numbers VERIFIED this session against base_commit 1535ac1.
// Clauses NOT listed here are UNCHANGED.
let device_host_eligible =
    // 4214 — WAS: group_spans.is_none(). Track R/W relax this behind a family gate.
    (group_spans.is_none() || ranking_device_covered(...) || pairwise_device_covered(...))
    // 4215 — WAS: ordered_learning_perm.is_none(). Track O (spike-gated).
    && (ordered_learning_perm.is_none() || ordered_device_covered(...))
    // 4225-4231 — CTR. Track K relaxes ONLY `col.projection.is_simple()` inside
    // ctr_types_are_device_covered (boosting.rs:2244); Track U relaxes the ctr_type
    // equality at :2245.
    && ( (materialized_ctr_features.is_empty()
            && structure_fold_columns.iter().all(Vec::is_empty))
         || (learning_folds_for_cycle == 1
             && one_hot_bins.is_empty()
             && ctr_types_are_device_covered(&materialized_ctr_features)) )
    // 4232 — Track N (spike-gated).
    && (!penalties_active || penalties_device_covered(...))
    // 4233 — Track M (spike-gated).
    && (params.monotone_constraints.is_empty() || monotone_device_covered(...))
    && matches!(params.grow_policy, SymmetricTree | Depthwise | Lossguide | Region)   // 4237-4243, unchanged
    // 4244-4246 — Track X (spike-gated).
    && (approx_dimension == 1 || multioutput_device_covered(...))
    && (!is_multiclass   || multioutput_device_covered(...))
    && (!is_multilabel   || multioutput_device_covered(...))
    // 4263-4270 — Track P removes the `&& grow_policy == SymmetricTree` restriction.
    && (matches!(params.bootstrap_type, No)
        || matches!(params.bootstrap_type, Bayesian | Bernoulli | Mvs | Poisson))
    && params.random_strength == 0.0                                                  // 4271, UNCHANGED (out of scope)
    // 4272 — Track V (spike-gated).
    && (eval_sets.is_empty() || eval_sets_device_covered(...))
    && has_any_scorable_feature(&matrix)                                              // 4288, unchanged
    && one_hot_cardinalities_fit_the_device(...)                                      // 4294-4299, unchanged
    // 4312 — Track A: REMOVED outright (the bias now rides DeviceTrainConfig.bias).
    // 4318 — Track E: WAS matches!(leaf_method, Gradient | Simple).
    && (matches!(params.leaf_method, LeafMethod::Gradient | LeafMethod::Simple)
        || (matches!(params.leaf_method, LeafMethod::Exact)
            && matches!(params.loss, Loss::Mae | Loss::Quantile { .. })));
```

```python
# bench/generator.py — EXTENDED (SPD-01). Existing `generate()` / `binary_target()`
# signatures and byte-output are UNCHANGED (D-06 single-source rule: the committed
# small-n correctness fixtures must stay bit-identical).
def generate_categorical(n_rows, n_cat_features, cardinalities, seed=42):
    """NEW. Seeded integer-coded categorical columns via numpy.random.RandomState
    (the SAME legacy Mersenne-Twister stream generate() uses, so the bytes are stable
    across numpy versions). Returns (n_rows, n_cat_features) int32."""

def generate_weights(n_rows, seed=42, kind="cycled"):
    """NEW. Non-uniform per-object sample weights. `kind="cycled"` yields the
    {0.5, 1.0, 2.0, 3.0} cycle the committed weighted_device_* fixtures already use,
    so the speed workload and the correctness fixtures share one weight recipe."""
```

## 5. Failure-isolated behavioral specifications

Every spec below is independently testable and has ONE primary failure cause. Grouped by
track; the wave/dependency ordering lives in `PLAN.md` §3.

Spike specs (Wave 4) are deliberately written as **goal + acceptance bar + CodeGraph-cited
starting points**, with NO fabricated Red/Green/Refactor: the exact device-kernel shape is
not knowable without the dedicated design pass the spike IS. This mirrors the prior
phase's Track O / T18 precedent.

---

### Track A — `boost_from_average` / non-zero bias (CR-01)

#### FPP-01 — Seed the resident approx from a per-fit `bias`
- Status: draft
- Principal failure reason: the device's resident approximant starts at zero instead of
  the fit's real starting approximant, so every tree trains against a wrong starting
  point.
- Scope: `crates/cb-backend/src/gpu_runtime/session.rs::GpuTrainSession::begin`,
  `crates/cb-compute/src/runtime.rs::DeviceTrainConfig`.
- Rationale: `session.rs:1578-1580` is a literal one-line hardcode —
  `let approx_h = upload_channel_floats(&client, &vec![0.0_f64; n]);` with the comment
  "The running approx starts all-zero (the RMSE-from-zero MVP; boost_from_average is out
  of scope…)". `starting_approx` (`boosting.rs:3641`) is a per-fit scalar. Upstream's
  `boost_from_average` default is `true` for RMSE and this project's `CatBoostBuilder`
  also defaults it `true` (`builder.rs:299`), so TODAY an unmodified default-config RMSE
  fit NEVER reaches the device. This is the highest-value-per-line fix in the phase.
- Input: `DeviceTrainConfig.bias: f64` (NEW plain-host field, default `0.0`).
- Output: `approx_h = upload_channel_floats(&client, &vec![config.bias; n])`.
- Dependencies: none.
- Given/When/Then:
  - Given `config.bias == 0.0`, when `begin` runs, then the uploaded resident approx is
    byte-identical to today's (D-04 regression: every currently-green device e2e test
    stays green unmodified).
  - Given `config.bias == b != 0.0`, when `begin` runs and the first tree is grown, then
    the resident der1 computed by `launch_der_binary_resident` from that seed equals the
    host reference `der1[i] = target[i] - b` (RMSE) / `target[i] - sigmoid(b)`
    (Logloss/CrossEntropy) at ε=1e-4.
- Invariants: the nonsym / Region / exact-leaf arms are UNAFFECTED — verified this
  session, they re-derive `der1` host-side from the CALLER's `approx`
  (`session.rs:1824-1844`, `:1863-1882`, `:1902-1932` via `host_der1`), and the caller's
  `approx` already starts at `vec![bias; n]` (`boosting.rs:3649-3660`). Only the
  OBLIVIOUS resident arm depends on the zero seed.
- Acceptance tests: a `cb-backend` self-oracle asserting the first-tree resident der1 for
  a non-zero bias matches the host reference at ε=1e-4, plus the byte-unchanged assertion
  for `bias == 0.0`.
- Out of scope: relaxing the eligibility clause (FPP-02).
- Traceability: `session.rs:1578-1580`; `runtime.rs:1162,1220` (`DeviceTrainConfig`
  defaults); `boosting.rs:3641` (`starting_approx`), `:4306-4312` (the clause's own
  comment naming CR-01 and the zero seed as the exact reason).
- Unresolved: none.

#### FPP-02 — Remove `device_host_eligible`'s `bias == 0.0` clause
- Status: draft
- Principal failure reason: a default-config `boost_from_average=true` fit is still
  refused the device path after FPP-01 made it correct.
- Scope: `crates/cb-train/src/boosting.rs:4312` + the `device_config` construction at
  `:4400-4460`.
- Preconditions: FPP-01 complete and green. (D-04 ordering discipline: never flip
  `device_host_eligible` before the underlying computation is verified correct —
  research.md pitfall #2.)
- Input: `bias: f64` (existing local, `boosting.rs:3641`).
- Output: the `&& bias == 0.0` clause is REMOVED, and `device_config.bias = bias` is set.
  Both changes land TOGETHER — relaxing the clause without populating the field would seed
  the device from zero on a biased fit, which is WRONG, not merely less good.
- Dependencies: FPP-01.
- Given/When/Then:
  - Given a default `boost_from_average=true` RMSE fit that is otherwise device-eligible,
    when `device_active` is checked, then it is `true` (proves the clause removal changed
    observable behaviour — the anti-false-pass pattern).
  - Given `boost_from_average=false` (`bias == 0.0`), when the same fit runs, then
    behaviour is byte-unchanged.
- Invariants: D-10-01 (one per-fit all-or-nothing decision) preserved.
- Acceptance tests: FPP-04's e2e fixture oracle is the primary proof; plus a fast
  gate-level `device_active == true` assertion via a `CountingGpu`-style runtime.
- Out of scope: `RmseWithUncertainty`'s per-dimension bias vector (multi-output, Track X).
- Traceability: `boosting.rs:4306-4312`, `:4400-4460`.
- Unresolved: none.

#### FPP-03 — New `bias_device_sym/` upstream fixture
- Status: draft
- Principal failure reason: no committed fixture trains with a non-zero starting
  approximant AND a device-eligible shape — verified this session: of the 33 fixtures
  carrying `boost_from_average`, only `feature_selection/`, `regression_skeleton/`,
  `model_sum/` and `ordered_boost_e2e/` pin it `true`, and none of those is
  device-eligible for other reasons.
- Input: a new per-directory `gen_fixtures.py` following the
  `crates/cb-oracle/fixtures/weighted_device_sym/` convention (verified: that is the
  current authoring pattern).
- Output: `crates/cb-oracle/fixtures/bias_device_sym/{gen_fixtures.py,config.json,X.npy,
  y.npy,borders.npy,model.json,predictions.npy}` produced by the INSTALLED
  `catboost==1.2.10`, `boost_from_average=True`, RMSE, `border_count=15` (16 bins — the
  device CTR/histogram width convention every existing device fixture uses),
  `bootstrap_type="No"`, `random_strength=0`, `leaf_estimation_method="Gradient"`,
  `score_function="L2"`, `grow_policy="SymmetricTree"`, `thread_count=1`.
- Dependencies: none (offline generation, parallel with FPP-01/02).
- Given/When/Then:
  - Given the generator runs, when it completes, then `config.json` records
    `"boost_from_average": true` and the recorded `mean(y)` is non-zero by more than
    `1e-6` (an anti-false-pass guard: a fixture whose bias happens to be ~0 could not
    discriminate FPP-01's fix).
  - Given the fixture is loaded, when compared against a `boost_from_average=False`
    sibling trained on the same `X`/`y`, then the predictions differ by more than `1e-6`
    (proves the fixture actually exercises the bias).
- Acceptance tests: a fixture smoke test asserting shapes, the non-zero bias, and the
  15-border float columns.
- Out of scope: non-SymmetricTree bias scenarios (covered by the device-vs-CPU ε=1e-4
  comparison in FPP-04, not by extra upstream generation).
- Traceability: `crates/cb-oracle/fixtures/weighted_device_sym/gen_fixtures.py`.
- Unresolved: none.

#### FPP-04 — E2e device oracle: non-zero-bias training
- Status: draft
- Principal failure reason: the biased device fit does not reproduce upstream at ≤1e-5.
- Preconditions: FPP-01, FPP-02, FPP-03.
- Output: a new `crates/cb-train/tests/device_bias_fit_test.rs` following the
  `device_weighted_fit_test.rs` precedent (module gated
  `#[cfg(any(feature = "rocm", feature = "cuda"))]`, local `CpuRefRuntime`, never a
  `cb_backend::CpuBackend` import, SKIP-by-printing off rocm/cuda).
- Given/When/Then:
  - Given the `bias_device_sym/` fixture, when trained via
    `cb_train::train(&GpuBackend::default(), ...)`, then `device_active == true`
    (assert the right arm fired — `CountingGpu` counting `grow_tree_on_device` calls
    equals `iterations`).
  - Given the same fit, when predictions are computed, then they match
    `predictions.npy` within ≤1e-5.
- Acceptance tests: this spec's own new test.
- Traceability: `crates/cb-train/tests/device_weighted_fit_test.rs`.
- Unresolved: none.

---

### Track E — Exact leaf on device (MAE / Quantile)

#### FPP-05 — Set `device_config.exact_leaf` + quantile α/δ from `params`
- Status: draft
- Principal failure reason: the fully-built, self-oracled device Exact order-statistic
  leaf path is never activated, because `device_config.exact_leaf` is never set anywhere
  in `cb-train`.
- Scope: `crates/cb-train/src/boosting.rs` device_config construction (`:4400-4460`).
- Rationale: `map_leaf_method` (`session.rs:996-1017`) returns
  `Some(DeviceLeafMethod::Newton)` whenever `!config.exact_leaf` — i.e. always, today —
  so `DeviceLeafMethod::Exact` and its `kernels/exact_quantile.rs` order-statistic kernel
  are structurally dead. Verified: `grep -n exact_leaf crates/cb-train/src/boosting.rs`
  shows only a COMMENT at `:4442` ("`exact_leaf` / `ctr` likewise stay default").
- Input: `params.leaf_method`, `params.loss` (for `Loss::Quantile { alpha, delta }`).
- Output: when `params.leaf_method == LeafMethod::Exact` AND
  `params.loss ∈ {Mae, Quantile}`, `device_config.exact_leaf = true` and
  `quantile_alpha`/`quantile_delta` are set from the loss (`Quantile`) or left at the
  `0.5`/`1e-6` MAE median defaults (`Mae`).
- Dependencies: none.
- Given/When/Then:
  - Given `leaf_method != Exact`, when the config is built, then `exact_leaf` stays
    `false` and the config is byte-identical to today (D-04).
  - Given `leaf_method == Exact` and `loss == Quantile { alpha: 0.7, delta: 1e-6 }`, when
    the config is built, then `exact_leaf == true`, `quantile_alpha == 0.7`,
    `quantile_delta == 1e-6`.
- Invariants: this spec alone must NOT change `device_active` for any fit — the
  `leaf_method` eligibility clause (`boosting.rs:4318`) still excludes `Exact` until
  FPP-06 lands. Setting the config first and relaxing the gate second is the mandated
  ordering (research.md pitfall #2: relaxing the gate without setting the config would
  silently commit MAE/Quantile fits to the Gradient `calc_average` leaf — WRONG, and worse
  than today's correct CPU fallback).
- Acceptance tests: a unit-level assertion on the constructed `DeviceTrainConfig` for both
  branches.
- Out of scope: the gate relaxation (FPP-06).
- Traceability: `boosting.rs:4442` (the comment to replace); `session.rs:980-1017`
  (`DeviceLeafMethod`, `map_leaf_method`); `runtime.rs:1162,1220`.
- Unresolved: none.

#### FPP-06 — Admit `LeafMethod::Exact` for the covered quantile family
- Status: draft
- Principal failure reason: an explicit `leaf_estimation_method="Exact"` MAE/Quantile fit
  is still refused the device path after FPP-05 made the config correct.
- Preconditions: FPP-05.
- Output: `boosting.rs:4318`'s clause gains an `Exact` arm gated to
  `matches!(params.loss, Loss::Mae | Loss::Quantile { .. })` — the EXACT intersection of
  (a) what the CPU validator permits and (b) what `map_leaf_method` covers.
- Dependencies: FPP-05.
- Given/When/Then:
  - Given `leaf_method == Exact` and `loss == Mae` (or `Quantile`), otherwise
    device-eligible, when `device_active` is checked, then it is `true`.
  - Given `leaf_method == Exact` and `loss == LogCosh`, when `device_active` is checked,
    then it is `false` — CPU permits Exact for LogCosh (`validate_leaf_method`,
    `boosting.rs:1388-1398`) but `map_leaf_method` returns `None` for it
    (`session.rs:1015`), so the device MUST decline rather than silently substitute a
    different leaf estimator. This is the discriminating negative test.
  - Given `loss == MultiQuantile`, when `device_active` is checked, then it is `false`
    (`approx_dimension == 1` still excludes it — MultiQuantile is Track X, not Track E).
  - Given `leaf_method ∈ {Gradient, Simple}`, when any existing device fit runs, then
    behaviour is byte-unchanged.
- Invariants: D-10-01 preserved.
- Acceptance tests: FPP-08's e2e oracle plus the LogCosh negative assertion.
- Out of scope: MAPE — verified this session that
  `validate_leaf_method` (`boosting.rs:1388-1398`) permits `Exact` only for
  `LogCosh | Mae | Quantile | MultiQuantile`, so a `Mape` + `Exact` fit is REJECTED
  CPU-side before any device consideration, and `map_leaf_method`'s `Loss::Mape` arm
  (`session.rs:1009-1013`) is unreachable from `cb_train::train`.
- Traceability: `boosting.rs:4313-4318`, `:1387-1411`; `session.rs:996-1017`.
- Unresolved: none.

#### FPP-07 — New `exact_leaf_device/` upstream fixture
- Status: draft
- Principal failure reason: no committed fixture is BOTH Exact-leaf AND device-shaped.
- Rationale: `crates/cb-oracle/fixtures/quantile_alpha05_mae/` and `quantile_alpha07/`
  DO pin `leaf_estimation_method="Exact"`, `boost_from_average=false`,
  `bootstrap_type="No"`, `random_strength=0`, `score_function="L2"`, depth 2, 4 float
  features, 50 rows — i.e. an almost device-shaped recipe — but they ship only
  `model.json` + `staged.npy` (no `predictions.npy`), they pin no `border_count` (so they
  inherit upstream's 254-border default rather than the 15-border/16-bin convention every
  existing device fixture uses), and their sole Rust consumer
  (`crates/cb-train/tests/wave3_quantile_loss_oracle_test.rs`) is a `CpuBackend`-importing
  file that cannot compile under `--no-default-features --features rocm`.
- Output: `crates/cb-oracle/fixtures/exact_leaf_device/{mae,quantile07}/` with the full
  `weighted_device_sym/`-shaped artifact set (`X.npy`, `y.npy`, `borders.npy`,
  `model.json`, `predictions.npy`, `config.json`, `gen_fixtures.py`), `border_count=15`.
- Given/When/Then:
  - Given the generator runs, when it completes, then `config.json` records
    `leaf_estimation_method="Exact"` and `border_count=15`, and `model.json` yields 4
    float-feature border lists of length 15.
  - Given the MAE scenario and the Quantile α=0.7 scenario, when their predictions are
    compared, then they differ by more than `1e-6` (proves α is load-bearing — an
    anti-false-pass guard against a device path that ignores `quantile_alpha`).
- Acceptance tests: a fixture smoke test.
- Traceability: `crates/cb-oracle/fixtures/quantile_alpha05_mae/config.json`,
  `weighted_device_sym/gen_fixtures.py`.
- Unresolved: whether the device's packed-cindex line width admits any `border_count`
  other than 15/16 bins — PLAN must confirm before pinning; 15 is chosen because it is
  the only value proven by existing green device fixtures.

#### FPP-08 — E2e device oracle: Exact-leaf training
- Status: draft
- Principal failure reason: the device Exact order-statistic leaf does not reproduce
  upstream's `CalculateWeightedTargetQuantile` at ≤1e-5.
- Preconditions: FPP-05, FPP-06, FPP-07.
- Output: a new `crates/cb-train/tests/device_exact_leaf_fit_test.rs`.
- Given/When/Then:
  - Given the `exact_leaf_device/mae` fixture, when trained on device, then
    `device_active == true` AND predictions match `predictions.npy` within ≤1e-5.
  - Given the `exact_leaf_device/quantile07` fixture, same bar — this is the case a
    device path that silently used `calc_average` instead of the order statistic FAILS,
    making it the discriminating test.
- Acceptance tests: this spec's own new test.
- Traceability: `crates/cb-backend/src/kernels/exact_quantile.rs` (the kernel being
  proven reachable).
- Unresolved: none.

---

### Track K — Combination / tensor CTR projections on device

#### FPP-09 — Admit non-simple projections and populate ALL projection members
- Status: draft
- Principal failure reason: a combination-CTR split is scored using only ONE of its member
  columns' bins, producing a WRONG (not merely worse) split.
- Scope: `crates/cb-train/src/boosting.rs::ctr_types_are_device_covered` (`:2241-2249`)
  and `build_device_ctr_config`'s `build_columns` closure (`:2292-2343`).
- Rationale: the device accumulation kernel already folds multi-column projections —
  `combine_projection_bins` (`crates/cb-backend/src/kernels/ctr_device.rs:89-117`) — and
  `build_ctr_cindex_columns` already calls it when `col.member_bins.len() != 1`
  (`session.rs:207-213`). The blocker is entirely `cb-train`-side: the gate requires
  `col.projection.is_simple()` (`:2244`) and the column builder extracts only
  `col.projection.cat_features().first()` into `member_bins: vec![member]` (`:2299`,
  `:2335`).
- Input: `col.projection.cat_features(): &[usize]` (all members, projection-sorted),
  `cat_eligible_buckets: &[Vec<u32>]`, `eligible_absolute: &[usize]`.
- Output: `ctr_types_are_device_covered` drops the `is_simple()` requirement;
  `build_columns` maps EVERY member through the `eligible_absolute` position lookup into
  `member_bins: Vec<Vec<u32>>` in projection-sorted order. Both land TOGETHER (research.md
  pitfall #3).
- Dependencies: none. (The two-permutation machinery already landed; `build_columns` is
  already called twice, for the structure and averaging halves — `boosting.rs:2345-2347` —
  so combination support is inherited by both automatically.)
- Given/When/Then:
  - Given a `max_ctr_complexity ≥ 2` fit whose materialized CTR columns include a
    2-member combination projection with all-Borders types, when
    `ctr_types_are_device_covered` is evaluated, then it returns `true`.
  - Given the same fit, when `build_device_ctr_config` runs, then each combination
    column's `member_bins.len()` equals `col.projection.cat_features().len()`, and each
    inner column is the raw bucket column of the corresponding member (never a repeat of
    the first).
  - Given a SIMPLE projection, when the same code runs, then `member_bins.len() == 1` and
    the resulting `DeviceCtrColumn` is byte-identical to today (D-04 regression — the
    already-green `device_ctr_fit_test.rs` must stay green unmodified).
- Invariants: the device's combined bin column must be INTEGER-IDENTICAL to the CPU's.
  Verified this session (see PLAN §1 V-4 for the full derivation): both sides fold members
  in projection-sorted order via the same `fold_cat_hash`
  (`crates/cb-train/src/projection.rs:84,191` vs `ctr_device.rs:75,106-110`) starting from
  `key = 0`, then remap keys to dense **first-seen bins in OBJECT order**
  (`crates/cb-train/src/ctr/ctr_feature.rs:207-220` vs `ctr_device.rs:102-116`). The CPU
  folds `calc_cat_feature_hash(value)` while the device folds
  `cb_data::perfect_hash_bins` codes (`boosting.rs:2105-2119`) — different KEYS, but both
  injective per feature, so the induced PARTITION and therefore the first-seen bin
  numbering are identical, and the read-before-increment prefix statistic
  (`ordered_ctr_prefix_kernel`, `ctr_device.rs:134-167`) depends only on bucket identity.
- Acceptance tests: a `cb-train`-level assertion that a 2-member projection yields
  `member_bins.len() == 2`; plus FPP-11's e2e oracle as the numeric proof.
- Out of scope: non-Borders CTR types (Track U); `permutation_count > 1`.
- Traceability: `boosting.rs:2241-2249`, `:2292-2343`; `ctr_device.rs:89-117`;
  `session.rs:197-230`.
- Unresolved: a 64-bit `fold_cat_hash` collision that occurs on ONE side but not the other
  would diverge the partitions. Practically negligible at fixture scale; documented as a
  residual in §9 and detected (not prevented) by FPP-11's ≤1e-5 bar.

#### FPP-10 — New `ctr_device_combo/` upstream fixture (mixed float + 2 cat)
- Status: draft
- Principal failure reason: no committed fixture is BOTH combination-CTR and
  device-reachable.
- Rationale: `crates/cb-oracle/fixtures/tensor_ctr_e2e/` and
  `ctr_mixed_simple_vs_combo/` exist but, like every pre-`ctr_device_mixed` CTR fixture,
  are cat-ONLY — and a cat-only pool yields `device_n_float == 0`, so
  `has_any_scorable_feature` and the session's own `n_features == 0` decline both fire
  (the prior phase's V-9 finding, still binding at `boosting.rs:4350-4351`).
- Output: `crates/cb-oracle/fixtures/ctr_device_combo/` modelled directly on
  `ctr_device_mixed/gen_fixtures.py` (the proven device-reachable CTR recipe) but with
  **two** categorical columns and `max_ctr_complexity=2`, so at least one materialized
  column carries a 2-member projection.
- Given/When/Then:
  - Given the generator runs, when it completes, then the trained `model.json` contains at
    least one CTR split whose projection has 2 members (assert in the generator; a fixture
    where upstream happened to choose only simple-projection splits cannot discriminate
    FPP-09 and must be re-seeded rather than accepted).
  - Given the fixture, when compared to a `max_ctr_complexity=1` sibling on the same
    `X`/`y`, then the predictions differ by more than `1e-6`.
- Acceptance tests: a fixture smoke test asserting the ≥2-member projection.
- Traceability: `crates/cb-oracle/fixtures/ctr_device_mixed/gen_fixtures.py`.
- Unresolved: none.

#### FPP-11 — E2e device oracle: combination-CTR training
- Status: draft
- Principal failure reason: the device's combined-projection CTR bins diverge from the
  CPU's, so leaf values (not just structure) miss upstream.
- Preconditions: FPP-09, FPP-10.
- Output: a new `crates/cb-train/tests/device_ctr_combo_fit_test.rs`.
- Given/When/Then:
  - Given the `ctr_device_combo/` fixture, when trained on device, then
    `device_active == true` and predictions match `predictions.npy` within ≤1e-5.
  - Given the same fixture trained on `CpuBackend`, when both prediction vectors are
    compared, then they agree within ε=1e-4 (device self-oracle bar, D-07).
- Acceptance tests: this spec's own new test.
- Traceability: `crates/cb-train/tests/device_ctr_fit_test.rs` (structural precedent).
- Unresolved: none.

---

### Track P — Non-symmetric / Region × bootstrap sampling

#### FPP-12 — Host sample multiplier for the nonsym and Region device growers
- Status: draft
- Principal failure reason: a Depthwise/Lossguide/Region fit with `bootstrap_type != No`
  scores splits over the UNSAMPLED derivatives, diverging from upstream's `sampledDocs`.
- Scope: `crates/cb-backend/src/gpu_runtime/session.rs::grow_one`'s Region and nonsym
  arms.
- Rationale: the prior phase's V-2 established (and this session re-confirmed the shape)
  that `grow_nonsym_tree` and `grow_region_tree` take `der1: &[f64]` / `weight: &[f64]` as
  plain host slices and use `der1` only as a contribution sum — so the weighted-der fix
  was a single CALLER-SIDE substitution per arm (`host_weighted_der1`). The bootstrap
  sample multiplier is the identical shape: substitute `der1[i] * sample[i]` and
  `weight[i] * sample[i]` into the SPLIT-SCORE inputs only, leaving the LEAF-estimation
  inputs unsampled — exactly the contract the `Runtime::grow_tree_on_device` doc already
  states for `sample` (`runtime.rs:1418-1423`: "the backend folds into the SPLIT-SCORING
  stat channels ONLY. Leaf estimation stays on the UNSAMPLED derivatives / weights").
- Input: `sample: &[f64]` (length 0 or `n`, existing seam parameter).
- Output: the nonsym and Region arms fold the multiplier into their split-score channels,
  mirroring the oblivious arm's existing `fold_weights_resident(&self.client,
  &weighted_der1_h, s, self.n)` treatment (`session.rs:1933-1976`).
- Dependencies: none. Reuse `cb_core::sum_f64` for any host reduction; do not hand-roll.
- Given/When/Then:
  - Given `sample.is_empty()`, when either arm runs, then behaviour is byte-unchanged
    (D-04 — `device_nonsym_fit_test.rs` / `device_region_fit_test.rs` stay green
    unmodified).
  - Given a length-`n` `sample` with some zeros, when a split is scored, then objects with
    `sample[i] == 0.0` contribute zero to BOTH the score der and the score weight (they
    drop out of the split histogram exactly as a `control == false` object drops out of
    upstream's `sampledDocs`).
  - Given the same length-`n` `sample`, when a LEAF value is computed, then it uses the
    UNSAMPLED `der1`/`weight` (matching the CPU reference and the seam's stated contract).
- Invariants: the split score and the unsplit-gain BASELINE must both consume the sampled
  channels — the prior phase's V-2 found three der-gather sites in `nonsym_grow.rs` and
  four in `region_device.rs`, all fixed by the one caller-side substitution.
- Acceptance tests: extend the existing `kernels::nonsym_grow_test.rs` /
  `kernels::region_device_test.rs` device-vs-CPU comparisons with a
  non-trivial-sample variant at ε=1e-4.
- Out of scope: SymmetricTree (already closed).
- Traceability: `session.rs:1682-1714` region/nonsym dispatch, `:1933-1976` (the oblivious
  precedent); `runtime.rs:1409-1430` (the `sample` contract).
- Unresolved: the exact current line numbers of the two `grow_one` dispatch arms — PLAN
  re-verifies (the prior phase's numbers predate the GDC merge).

#### FPP-13 — Relax the bootstrap × grow_policy cross-product clause
- Status: draft
- Principal failure reason: a sampled Depthwise/Lossguide/Region fit is still refused the
  device path after FPP-12 made it correct.
- Preconditions: FPP-12.
- Output: `boosting.rs:4263-4270`'s inner `&& matches!(params.grow_policy,
  EGrowPolicy::SymmetricTree)` restriction is removed for the host-sampled types
  (`Bayesian`/`Bernoulli`/`Mvs`). **Poisson stays SymmetricTree-only**: it is the one arm
  the DEVICE samples resident (`sample_from_host = false`, `boosting.rs:4424-4426`), the
  resident sampler lives in the oblivious path, and `boosting.rs:4497+` hard-errors a
  Poisson fit that fails to commit — widening it without a resident nonsym sampler would
  turn a correct CPU fallback into an error.
- Dependencies: FPP-12.
- Given/When/Then:
  - Given `grow_policy=Depthwise, bootstrap_type=Bernoulli`, when `device_active` is
    checked, then it is `true`.
  - Given `grow_policy=Region, bootstrap_type=Poisson`, when `device_active` is checked,
    then it is `false` AND the fit surfaces the existing typed Poisson error rather than
    silently falling back — the discriminating negative test.
- Acceptance tests: FPP-14; plus the Poisson negative assertion.
- Traceability: `boosting.rs:4247-4270`, `:4415-4427`, `:4497+`.
- Unresolved: none.

#### FPP-14 — E2e device oracle: sampled non-symmetric training
- Status: draft
- Principal failure reason: a sampled nonsym/Region device fit diverges from the CPU
  reference (the RNG draw ORDER, not just the multiplier, is the usual failure).
- Preconditions: FPP-12, FPP-13.
- Output: a new `crates/cb-train/tests/device_nonsym_bootstrap_test.rs`, following
  `device_bootstrap_parity_test.rs`'s `CountingGpu` + per-scenario structure.
- Given/When/Then:
  - Given `{Depthwise, Lossguide, Region} × {Bayesian, Bernoulli, Mvs}`, when each is
    trained on device and on `CpuBackend` from the same seed, then the predictions agree
    within ε=1e-4 AND the `CountingGpu` device-tree count equals `iterations` (no silent
    mid-run fallback).
- Acceptance tests: this spec's own new test.
- Traceability: `crates/cb-train/tests/device_bootstrap_parity_test.rs`.
- Unresolved: whether `replay_grow_draws` (`boosting.rs:5261-5266`) — which restores the
  RNG phase the device branch skipped — is correct for the non-symmetric level search,
  whose candidate count differs from the oblivious one. PLAN must verify; if it is not,
  this becomes the spec's principal failure reason and FPP-14 gains a draw-accounting
  sub-assertion.

---

### Track S — The single Phase-13 grow-seam extension

#### FPP-15 — Extend `Runtime::grow_tree_on_device` with `family: Option<&FamilyTreeArgs>`
- Status: draft
- Principal failure reason: four families each need a different per-tree descriptor and,
  without ONE shared seam extension, four overlapping trait-signature changes collide on
  the same call sites (research.md pitfall #7).
- Scope: `crates/cb-compute/src/runtime.rs` (trait + new enum),
  `crates/cb-backend/src/gpu_backend.rs`, `crates/cb-train/src/boosting.rs`, and every
  test wrapper/mock.
- Preconditions: none — pure trait/type change, startable immediately, independent of
  every other track.
- Input/Output: see §4 Typed contracts (`FamilyTreeArgs<'a>`, the new
  `family: Option<&FamilyTreeArgs>` parameter).
- Dependencies: none.
- Given/When/Then:
  - Given the trait DEFAULT implementation, when called with `family` either `Some` or
    `None`, then it still returns `Ok(None)` unconditionally (byte-unchanged — the new
    parameter is bound-and-ignored, exactly like every existing parameter).
  - Given `GpuBackend`'s override called with `family: None` (every currently-green test),
    then behaviour is byte-unchanged.
  - Given every other call site in the workspace, when the signature changes, then the
    workspace compiles with `None` threaded through everywhere, and **every test wrapper
    forwards the parameter verbatim** (`self.inner.grow_tree_on_device(approx, target,
    sample, family)`) rather than hard-coding `None` — a wrapper that swallows it would
    silently drop family state once Wave 4 makes it load-bearing.
- Invariants: `FamilyTreeArgs` is plain host types only; no `cubecl` and no `cb-train`
  type may appear on it (T-10-04).
- Acceptance tests: `cargo check -p cb-compute`, `cargo check -p cb-backend
  --no-default-features --features rocm`, `cargo test --workspace` all green; plus a new
  sibling `runtime_family_args_test.rs` constructing each variant from primitive slices.
- Out of scope: actually CONSUMING the descriptor (Tracks R/W/X, Wave 4).
- Traceability: `runtime.rs:1430-1437` (the trait method to extend);
  `gpu_backend.rs:310` (the one real override); `boosting.rs:4772` (the one production
  caller).
- Unresolved: the exact site COUNT — research.md quoted the prior phase's "10 sites", but
  this session's `grep -rn "grow_tree_on_device" crates/ --include=*.rs` finds more test
  files have landed since. PLAN §1 pins the current, re-counted list; the Red step
  enumerates it and STOPS if the compiler reports a site not on that list.

---

### Track T — `task_type` (D1)

#### FPP-16 — `task_type` as an informational/validated parameter
- Status: draft
- Principal failure reason: `task_type` is either (a) still rejected as a parity gap,
  blocking every drop-in CatBoost migration that passes it, or (b) silently accepted as a
  no-op, violating this project's own Honesty policy.
- Scope: `crates/catboost-rs-py/src/params.rs`.
- Rationale: D1. Today `status_of("task_type") == KnownNotYet` (`VOCABULARY:233`, absent
  from `IMPLEMENTED:66-128`), so `validate_params` (`:395-423`) rejects it at `fit()`.
  Backend selection is compile-time only (`builder.rs:1183-1190`), which `CLAUDE.md`
  locks. `catboost-rs-py` already carries the `cpu`/`rocm`/`wgpu`/`cuda` feature
  passthrough, so the compiled-backend question is answerable in-crate with a `cfg!`.
- Input: the `task_type` kwarg, a Python string.
- Output:
  - `"CPU"` (any case) → ACCEPTED, no behavioural change. Justification recorded in the
    param doc: every device-ineligible config already falls back to the byte-unchanged CPU
    grower (D-04), so a CPU-feature wheel and a GPU-feature wheel both honour the request.
  - `"GPU"` → ACCEPTED iff compiled with `wgpu`/`cuda`/`rocm`; on a `cpu`-only wheel it
    raises `CatBoostParameterError` naming the required build
    (e.g. "`task_type=\"GPU\"` requires a wheel built with `--features cuda` (or `rocm` /
    `wgpu`); this wheel was built with `--features cpu`").
  - Any other value → `CatBoostParameterError` listing the two legal values.
  - `None` → inert (upstream's universal "not set", handled by the existing `get()`
    convention).
  - `task_type` moves from `KnownNotYet` to `IMPLEMENTED`.
- Dependencies: none — independently mergeable (D7 Wave 2).
- Given/When/Then:
  - Given `task_type="CPU"` on any wheel, when `fit()` runs, then it completes and the
    resulting model is identical to the same fit without the kwarg.
  - Given `task_type="GPU"` on a `cpu`-only wheel, when `fit()` runs, then it raises
    `CatBoostParameterError` whose message names the required Cargo feature — NOT a
    silent no-op, NOT a generic "parity gap".
  - Given `task_type="GPU"` on a `--features cuda` wheel, when `fit()` runs, then it
    completes; `device_host_eligible` and every gate below it behave EXACTLY as they do
    without the kwarg (this is pure input validation — assert by comparing predictions
    with and without the kwarg for bit equality).
  - Given `task_type="TPU"`, when `fit()` runs, then it raises `CatBoostParameterError`
    listing `CPU`/`GPU` — not a Levenshtein "unknown parameter" suggestion, since the
    parameter NAME is known.
- Invariants: `devices` and `device_config` (`params.rs:234-235`) stay `KnownNotYet` —
  this spec does not touch multi-device selection.
- Acceptance tests: Rust unit tests in the existing `params_test.rs` sibling covering all
  five branches, plus a Python-level pytest asserting the bit-identical-predictions claim
  and the error message text.
- Out of scope: any change to device dispatch behaviour.
- Traceability: `params.rs:3-18` (Honesty policy), `:66-128`, `:233-235`, `:395-423`;
  `builder.rs:1183-1190`; `crates/catboost-rs-py/Cargo.toml` `[features]`.
- Unresolved: whether `task_type` should also be echoed by `get_all_params()`/
  `get_params()` — PLAN to check whether those surfaces enumerate `IMPLEMENTED` and, if
  so, whether adding `task_type` perturbs any existing pinned-params test.

---

### Track G — Cross-gap composition and regression guards

#### FPP-17 — Cross-gap composition regression tests
- Status: draft
- Principal failure reason: one relaxation silently widens an UNRELATED existing
  exclusion, admitting a combination no kernel implements.
- Preconditions: every Wave-1 gate relaxation (FPP-02, FPP-06, FPP-09, FPP-13).
- Output: a new `crates/cb-train/tests/device_fpp_composition_test.rs` (separate from the
  prior phase's `device_gate_composition_test.rs`, which stays untouched) proving:
  1. bias × weighted × CTR together correctly ADMITS (all three are independently correct;
     nothing about their union is unimplemented) — a POSITIVE composition test.
  2. Exact-leaf × CTR still DECLINES (`session.rs`'s own exact × CTR exclusion is
     unchanged — the device declines rather than combining two leaf mechanisms).
  3. Exact-leaf × sampling still DECLINES (same reasoning).
  4. Combination-CTR × one-hot still DECLINES (SPEC-OH-26, untouched).
  5. `permutation_count > 1` × combination-CTR still DECLINES (the
     `learning_folds_for_cycle == 1` guard, untouched).
  6. An already-excluded combination unrelated to this phase (`random_strength != 0`)
     is STILL excluded.
- Given/When/Then: one per numbered item, each asserting the OBSERVABLE
  (`device_active`, via a `CountingGpu`-style runtime) rather than a specific clause — the
  prior phase's V-11 lesson that a decline may legitimately fire at either the host gate or
  the backend gate.
- Invariants: this spec requires NO new production code. Any failing assertion is a bug in
  the owning relaxation spec, not here.
- Acceptance tests: this spec's own new test file.
- Traceability: `crates/cb-train/tests/device_gate_composition_test.rs` (precedent).
- Unresolved: none.

---

### Track O — Ordered boosting on device (D2: spike → implement)

#### FPP-18 — SPIKE: per-segment ordered split-score device kernel design note
- Status: draft (spike — NO Red/Green/Refactor; the deliverable is a design note)
- Principal failure reason (of the spike itself): the design note fails to produce a
  concrete, sizeable kernel shape, so the implementation specs below cannot be refined
  into executable tasks.
- **Goal**: produce a short (≤2 page) kernel-design note answering, with citations:
  1. What exactly does CPU Ordered do that Plain does not? (Starting point:
     `crates/cb-train/src/tree.rs:2383` `score_candidate_ordered`, `:2427`
     `select_level_ordered`, `:2498` `greedy_tensor_search_oblivious_ordered` — verified
     present at these lines this session. For EVERY candidate `(feature, border)` at EVERY
     level it re-assigns leaves over the whole candidate set and sums `l2_split_score`
     across `body_tail_segments(n, fold_len_multiplier)`
     (`crates/cb-train/src/fold.rs:137`), each with its own `scale_l2_reg(l2,
     body_sum_weight, body_finish)` (`fold.rs:368` `body_sum_weights`).)
  2. Can the existing resident histogram substrate be driven PER SEGMENT, or is a new
     `#[cube]` kernel genuinely required? (Starting point: the existing per-level
     fill/score/split shape in `crates/cb-backend/src/gpu_runtime/mod.rs`
     `grow_oblivious_tree_resident`, and the host-driven
     `crates/cb-backend/src/kernels/nonsym_grow.rs` / `region_device.rs` precedent where
     the HOST controls the level loop and the device scores candidates.)
  3. What is the segment count at realistic `n`, and does `n_segments` independent
     partial-sum reductions per candidate fit the launch budget the D-05 "O(1) per-level
     crossings only" rule permits?
  4. What must cross the seam? (`FamilyTreeArgs` has no Ordered variant in §4 by design —
     the spike decides whether one is needed, or whether the per-fit-constant boundaries
     can ride `DeviceTrainConfig` instead. The prior phase's B-2 finding stands:
     `body_tail_boundaries(n, multiplier)` is a pure function of `(n, multiplier)`, i.e.
     **per-FIT constant**, not per-tree dynamic.)
  5. Does the ordered path support any score function other than L2? (`score_candidate_
     ordered` hard-codes `l2_split_score`; if so, decline every other score function to
     CPU rather than approximating.)
- **Acceptance bar**: the note names a concrete kernel (or a concrete reuse of an existing
  one), a launch shape, a seam decision, and a size estimate in net-new `#[cube]` lines
  plus host-driver lines.
- **ESCALATION GATE (mandatory, D2)**: if the spike concludes the work exceeds
  **~3 days-equivalent of net-new kernel work** — the prior phase's own yardstick was
  "comparable in size to the ENTIRE CTR-averaging-permutation deliverable" — then **STOP,
  write the note, and report back to the user before writing or executing any
  implementation task.** Do not silently proceed.
- Dependencies: none (read-only analysis). Consult
  `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` BEFORE any kernel
  sketch (AGENTS.md mandate).
- Out of scope: writing any production code.
- Traceability: `tree.rs:2383,2427,2498`; `fold.rs:105,137,368`;
  `session.rs:1073-1076` (the unconditional decline);
  `../gpu-device-coverage-expansion/PLAN.md` V-7 (the finding this spike supersedes).

#### FPP-19 — Ordered per-segment split scoring on the device grow path
- Status: draft (goal + acceptance bar only — FPP-18's output refines this into executable
  steps; writing line-precise steps today would be fabrication)
- Principal failure reason: the device Ordered arm chooses different splits from
  `greedy_tensor_search_oblivious_ordered`.
- **Goal**: for a fixed input, the device Ordered arm chooses the SAME per-level
  `(feature, border)` as the CPU reference.
- **Acceptance bar**: integer equality on the chosen `(feature, border)` at every level
  against a transcribed CPU reference on a frozen small fixture, and ε=1e-4 on the summed
  per-segment score.
- Preconditions: FPP-18 complete and its escalation gate cleared; FPP-15 if the spike
  decides an Ordered `FamilyTreeArgs` variant is needed.
- Out of scope: Ordered × CTR (already mutually exclusive both host-side,
  `boosting.rs:5443-5444`-equivalent, and backend-side — `map_ordered_coverage`
  (`session.rs:585+`) requires `config.ctr.is_none()`).

#### FPP-20 — Consume `map_ordered_coverage` and relax the Ordered clause
- Status: draft (goal + acceptance bar only)
- Principal failure reason: a covered Ordered fit still declines at `session.rs:1073-1076`
  and/or `boosting.rs:4215` after FPP-19 made the grow correct.
- **Goal**: `begin()` stops discarding `map_ordered_coverage`'s result; `boosting.rs:4215`
  admits a covered Ordered fit, guarded so Ordered × CTR never composes.
- **Acceptance bar**: a covered Ordered fit reaches the device (`device_active == true`);
  an uncovered one still declines; Ordered × CTR still declines.
- Note: the existing `begin_declines_ordered_to_cpu` self-test
  (`crates/cb-backend/src/gpu_runtime/ordered_test.rs`) is an INTENTIONAL update site —
  it must be flipped, never weakened, and its negative case must be preserved.

#### FPP-21 — New `ordered_boost_device/` fixture + e2e oracle
- Status: draft (goal + acceptance bar only)
- Principal failure reason: no reusable Ordered fixture is device-eligible — the prior
  phase's V-8 verified `crates/cb-oracle/fixtures/ordered_boost_e2e/config.json` pins
  `"boost_from_average": true` (re-confirmed this session, one of only 4 fixtures that
  do). **Note**: FPP-02 removes the `bias == 0.0` clause, so after Wave 1 this fixture may
  become reusable after all — PLAN must re-evaluate rather than pre-generate.
- **Goal**: an Ordered device fit matches upstream at ≤1e-5 over all iterations.

---

### Track U — Buckets / BinarizedTargetMeanValue / Counter CTR on device (D4.1: spike → implement)

#### FPP-22 — SPIKE: non-Borders CTR accumulation semantics on device
- Status: draft (spike)
- **Goal**: a design note answering, with citations, what each of the three remaining
  CPU-legal CTR types actually accumulates, and whether each can reuse
  `ordered_ctr_prefix_kernel`'s resident scratch shape or needs its own.
  Starting points, all verified present:
  - The authoritative statement of the gap: `ctr_types_are_device_covered`'s doc comment
    (`boosting.rs:2230-2240`) — "the device CTR arm implements the ordered binclf
    `(good + prior) / (total + 1)` statistic over SIMPLE projections… the device kernels do
    not implement those accumulation semantics."
  - CPU reference per type: `crates/cb-train/src/ctr/ctr_feature.rs:266-330` (the per-type
    dispatch producing `(nums, denoms, ctr_value, quantize_in_f32)`),
    `crates/cb-train/src/ctr/online.rs` (`TCtrHistory` class counts, `TCtrMeanHistory`
    float target sums, Counter total counts).
  - The device primitive to extend or fork: `ctr_device.rs:134-167`
    (`ordered_ctr_prefix_kernel`, resident `counts: &mut Array<u32>` scratch,
    read-before-increment).
  - The `f32`-domain trap: `TCtrMeanHistory::Sum` is `f32` upstream
    (`online_ctr.h:373`, cited in `bake.rs`), and `ctr_feature.rs`'s dispatch carries a
    `quantize_in_f32` flag — a BTMV/Counter device kernel that accumulates in f64 will
    miss ≤1e-5. This is the single highest-risk detail in the track.
  - `target_border_idx != 0` (Buckets emits 2 columns at binclf,
    `boosting.rs:2131-2134`) — today's gate pins `target_border_idx == 0`
    (`boosting.rs:2246`), so Buckets support requires the multi-target-border column
    dimension too, not just a new accumulator.
- **Acceptance bar**: the note states, per type, whether it is (a) a parameter change to
  the existing kernel, (b) a sibling kernel sharing the scratch shape, or (c) a genuinely
  new resident design; with a size estimate for each.
- **ESCALATION GATE**: if any of the three lands in category (c) at more than
  ~2 days-equivalent, STOP and report before writing its implementation task.
- Note (D4): NO shared-infrastructure assumption may be carried between this track and
  Tracks M/N — research.md's own LOW-confidence finding.

#### FPP-23 — Implement the non-Borders CTR device accumulators
- Status: draft (goal + acceptance bar only; FPP-22 refines)
- **Goal**: `ctr_types_are_device_covered` admits Buckets / BinarizedTargetMeanValue /
  Counter (under `counter_calc_method=SkipTest`), and the device produces bit-identical
  binarized cindex columns to the CPU for each.
- **Acceptance bar**: a per-type device-vs-CPU bit-exactness self-oracle (the existing
  `ctr_binarized_cindex_column_bit_exact` precedent), plus an e2e ≤1e-5 oracle per type
  against a NEW mixed float+cat fixture — the existing `ctr_buckets_simple/`,
  `ctr_btmv_simple/`, `ctr_counter_simple/` fixtures are cat-only and device-unreachable.

---

### Track M — `monotone_constraints` on device (D4.2: spike → implement)

#### FPP-24 — SPIKE: monotone-constrained split scoring on device
- Status: draft (spike)
- **Goal**: a design note establishing where monotonicity enters the CPU split-candidate
  evaluation and whether the device scorer can be gated the same way.
  Starting points: `params.monotone_constraints: Vec<i8>` (`builder.rs:177-182`,
  `monotone_constraints_default()`), the eligibility clause `boosting.rs:4233`, and the
  CPU consumption sites (`grep -n monotone crates/cb-train/src/` — PLAN pins them).
  Research.md verified via exhaustive grep that `crates/cb-backend/src/` has **zero**
  non-coincidental hits for "monotone" — there is no device precedent whatsoever.
- **Acceptance bar**: the note states whether the constraint is a candidate FILTER (cheap:
  a host-side mask over the candidate set, no kernel change) or a per-leaf VALUE clamp
  (expensive: touches the leaf reduce), with a size estimate.
- **ESCALATION GATE**: if it is not a candidate filter, STOP and report before writing the
  implementation task.

#### FPP-25 — Implement monotone constraints on the device grow
- Status: draft (goal + acceptance bar only; FPP-24 refines)
- **Goal**: `boosting.rs:4233` admits a monotone-constrained fit; the device chooses the
  same splits and produces the same leaf values as the CPU reference.
- **Acceptance bar**: device-vs-CPU ε=1e-4 on a monotone-constrained fit, plus an e2e
  ≤1e-5 oracle against a NEW upstream fixture (none exists — verified).

---

### Track N — Feature-weight penalties on device (D4.3: spike → implement)

#### FPP-26 — SPIKE: feature-weight penalties on device
- Status: draft (spike)
- **Goal**: a design note covering all four penalty knobs
  (`feature_weights`, `first_feature_use_penalties`, `per_object_feature_penalties`,
  `penalties_coefficient` — all four are IMPLEMENTED CPU-side,
  `crates/catboost-rs-py/src/params.rs:112-116`) and how each modifies split-candidate
  evaluation.
  Starting points: `penalties_active` (`boosting.rs:3780-3782`), its consumption at
  `:3211`, `:3254`, `:4595`, `:5853`, `:5918`, and the eligibility clause `:4232`.
  Note `feature_weights` is a MULTIPLICATIVE gain weight (`builder.rs:164-166`) while the
  two penalty vectors are ADDITIVE (`:168-172`) — they may not share a device shape.
- **Acceptance bar**: the note states, per knob, whether it is a per-candidate scalar
  multiply/subtract the device scorer can apply from an uploaded per-feature vector
  (cheap) or requires per-object state (`per_object_feature_penalties` is the suspect),
  with a size estimate.
- **ESCALATION GATE**: if `per_object_feature_penalties` requires per-object device state,
  STOP and report; consider splitting the track so the three cheap knobs land without it.

#### FPP-27 — Implement feature-weight penalties on the device grow
- Status: draft (goal + acceptance bar only; FPP-26 refines)
- **Goal**: `boosting.rs:4232` admits a penalised fit for whichever subset FPP-26 finds
  tractable; the device split choice matches the CPU reference.
- **Acceptance bar**: integer-equal split choices vs CPU on a frozen fixture, plus an e2e
  ≤1e-5 oracle against a NEW upstream fixture.

---

### Track V — `eval_sets` on device (D5: spike)

#### FPP-28 — SPIKE: eval sets + overfitting detector on the device branch
- Status: draft (spike)
- **Goal**: a design note confirming or refuting tractability.
  This session's verification already narrowed it substantially and the spike should
  START from these findings rather than re-derive them:
  - The eval-set update is `last_tree_eval_contribution(&trees, &non_symmetric_trees,
    &region_trees, em, obj)` (`boosting.rs:2511-2527`) — it reads the FOLDED tree
    ensembles, **not** an approx buffer or any device state.
  - The device branch already folds its grown tree into exactly those three ensembles
    (`boosting.rs:4787-4800`).
  - The device branch then `continue`s at `boosting.rs:5275`, skipping the entire CPU body
    including the `if has_test { … }` block at `:6690-6733`.
  ⇒ The mechanical shape is "run the same `if has_test` block before the device branch's
  `continue`". The spike must determine whether that is sufficient, specifically:
  1. `use_best_model` truncation and the detector's `break` — do they interact correctly
     with the device session's `DeviceSessionGuard` teardown (`boosting.rs:4489-4492`)?
  2. Snapshot/resume interaction on the device branch.
  3. A CTR device fit's eval matrix has no CTR columns — does `tree_eval_contribution`
     handle a `LevelKind::Ctr` tree at all, or must eval_sets × CTR stay excluded?
- **Acceptance bar**: the note answers all three, and states whether the change is
  ≤~50 lines in `boosting.rs` with no kernel work (the hypothesis) or something larger.
- **ESCALATION GATE**: if the answer to (3) forces a device-side eval CTR materialization,
  STOP and report — that is a new kernel track, not a wiring change.

#### FPP-29 — Implement eval sets on the device branch
- Status: draft (goal + acceptance bar only; written in detail ONLY if FPP-28 confirms)
- **Goal**: `boosting.rs:4272` admits a fit with eval sets; the device branch updates
  `eval_approx`, computes the eval metric, and drives the overfitting detector /
  `use_best_model` identically to the CPU branch.
- **Acceptance bar**: a device fit with an eval set and `od_type=Iter` stops at the SAME
  iteration as the CPU fit from the same seed, and both produce predictions agreeing at
  ε=1e-4.

---

### Tracks W / R / X — Phase-13 per-family grow consumption (spike → implement)

Each of these consumes FPP-15's seam. Each gets its own spike because research.md's
"Landed / Missing" split shows the missing half is NOT wiring:

#### FPP-30 — SPIKE: pairwise grow consumption (`PairLogitPairwise`, `YetiRankPairwise`)
- Status: draft (spike)
- **Goal**: a design note sizing the missing batched Cholesky SOLVE and the per-tree
  pair/group descriptor. Starting points: `map_pairwise_coverage` (`session.rs:373-409`),
  the decline at `session.rs:1091-1107` (which explicitly names "the batched Cholesky
  SOLVE + the per-tree pair/group descriptor seam … land in Plan 02 (GPUT-21)"), the
  landed `launch_pairwise_assemble_system_into`, `cb_compute::is_pairwise_scoring`
  (`crates/cb-compute/src/ranking_der.rs:46-51`), and the CPU per-tree adjacency re-sample
  at `boosting.rs:5278-5281`.
- **Acceptance bar**: the note sizes the Cholesky solve (batched, per-leaf, SPD) and
  states whether it reuses any existing device linear-algebra primitive.
- **ESCALATION GATE**: report before implementation regardless of size — this is the one
  family whose missing half is explicitly named as a whole unexecuted plan (GPUT-21).

#### FPP-31 — SPIKE: ranking grow consumption (`QueryRMSE`, `QuerySoftMax`, `YetiRank`)
- Status: draft (spike)
- **Goal**: a design note sizing the per-tree query-descriptor grow. Starting points:
  `map_ranking_coverage` (`session.rs:460-504`), the decline at `session.rs:1127-1134`,
  the landed `crate::gpu_runtime::ranking` der driver (self-oracled against
  `cb_compute::calc_ders_for_queries`), and `boosting.rs:4214`'s `group_spans.is_none()`
  first clause. Note the THIRD layer of gap: `QueryCrossEntropy` has no `Loss` variant at
  all and is independently gated off at the coverage-map level.
- **Acceptance bar**: the note states whether `FamilyTreeArgs::Ranking { group_offsets }`
  as specced in §4 is sufficient, or whether the per-tree YetiRank adjacency re-sample
  (`boosting.rs:5278-5281`) forces more across the seam.

#### FPP-32 — SPIKE: multi-output grow consumption
- Status: draft (spike)
- **Goal**: a design note sizing the shared multi-dim grow. Starting points:
  `map_multiclass_coverage` (`session.rs:532-556`), the decline at `session.rs:1149-1157`,
  the landed `grow_multiclass_block` K-dim Newton block-leaf driver (coupled softmax vs
  diagonal separable), `DeviceGrownTree.leaf_values`' EXISTING `leaf_count × approx_dim`
  row-major block contract (`runtime.rs`, so the return type needs no change), and the
  three eligibility clauses at `boosting.rs:4244-4246`. Note the FOURTH layer:
  `MultiQuantile`'s exact-quantile leaf is not reached by `map_multiclass_objective`.
- **Acceptance bar**: the note states whether `FamilyTreeArgs::MultiOutput { approx_k,
  approx_dim }` as specced in §4 is sufficient, and how the device-grown K-dim tree folds
  into cb-train's multi-output apply path.

#### FPP-33 / FPP-34 / FPP-35 — Implement pairwise / ranking / multi-output grow consumption
- Status: draft (goal + acceptance bar only; the corresponding spike refines each)
- **Goal** (each): the family's fits reach the device and match upstream at ≤1e-5 on a NEW
  per-family fixture (none of the existing fixtures covers any Phase-13 family end-to-end
  on device).
- **Acceptance bar** (each): `device_active == true`, predictions ≤1e-5 vs upstream, and
  every OTHER family still declines (D-10-01 all-or-nothing per family).

---

### Track SPD — Kaggle P100 speed supremacy (Part 2)

#### SPD-01 — Extend the workload generator with categoricals and non-uniform weights
- Status: draft
- Principal failure reason: the benchmark cannot exercise the CTR or weighted axes at all,
  because `bench/generator.py` produces only float32 gaussian features and a
  linear-plus-noise / logistic-thresholded target — verified this session: the module has
  `generate()` and `binary_target()` and nothing else feature-shaped.
- Input/Output: see §4's `generate_categorical` / `generate_weights` contracts.
- Dependencies: none. numpy only (the module's own stated "dependency-light" constraint).
- Given/When/Then:
  - Given the existing `generate(n_rows, n_features, seed)` is called with the committed
    `CORRECTNESS_CONFIG`, when its output is hashed, then the hash is UNCHANGED from
    before this spec (the D-06 single-source rule — the committed small-n correctness
    fixtures must not drift).
  - Given `generate_categorical` is called twice with the same seed, when the outputs are
    compared, then they are byte-identical (legacy `RandomState`, stable across numpy
    versions — the module's own stated reason for not using `default_rng`).
  - Given `generate_weights(n, kind="cycled")`, when inspected, then the values cycle
    `{0.5, 1.0, 2.0, 3.0}`, matching `weighted_device_sym/gen_fixtures.py`'s recipe so the
    speed workload and the correctness fixtures share one weight definition.
- Acceptance tests: a local (non-Kaggle) `python bench/generator.py`-level self-check
  asserting the unchanged hash for the existing configs.
- Traceability: `bench/generator.py` (`CORRECTNESS_CONFIG`, `SPEED_CONFIG`, `generate`,
  `binary_target`).
- Unresolved: none.

#### SPD-02 — Grid benchmark harness (one Kaggle session, D3)
- Status: draft
- Principal failure reason: the benchmark reports a speedup ratio that run-to-run noise
  could flip, or silently measures a CPU fallback as if it were a device fit.
- Input: the D3 grid — `{SymmetricTree, Depthwise} × {RMSE, Logloss} × {uniform,
  non-uniform weights} × {no CTR, single-permutation Borders CTR where the shape allows
  float+cat} × 2 dataset shapes at/above `n = 100_000`` (`bench/RESULTS.md` D-10-09's
  recorded crossover) ≈ 16–24 cells, each run **3×**.
- Output: `bench/full_param_gpu_speed/bench.py` — a NEW harness, not an edit of
  `bench/quick_gpu_speed/bench.py` (which stays as the frozen record of the r4a/b/c runs)
  — preserving verbatim:
  - the `timed_fit` pattern: UNTIMED warm fit → TIMED fit → `predict()` on a slice to
    drain the lazy CubeCL queue before stopping the clock (`bench.py:315-336`);
  - `build_eligibility_audit()` (`bench.py:38`) — the static precondition checklist,
    EXTENDED with the newly-reachable Wave-1 axes (bias, exact leaf, combination CTR,
    sampled non-symmetric);
  - the always-included honesty caveat that device activation is not observable from
    Python, so a silent CPU fallback cannot be 100% ruled out;
  - the single `maturin build --release --no-default-features --features cuda` build
    (`bench.py:220-225`), executed ONCE for the whole grid.
- Dependencies: SPD-01.
- Given/When/Then:
  - Given a cell, when it is run, then the harness records THREE independent timed fits
    per arm and reports the median plus min/max — never a single number.
  - Given a cell whose `catboost_rs` arm raises, when the run completes, then the failure
    is recorded in `result["timings"]["errors"]` and the cell is reported as FAILED —
    never silently dropped from the summary (the existing harness's own discipline).
  - Given the whole grid, when it completes, then the total wall clock fits inside ONE
    Kaggle GPU session (D3) — the harness must print a running elapsed budget and, on
    overrun, skip remaining cells with an explicit "BUDGET EXCEEDED, N cells not run"
    record rather than being killed mid-write.
  - Given a cell configured with a `boost_from_average=True` / Exact-leaf /
    combination-CTR / sampled-Depthwise recipe (the Wave-1 wins), when
    `build_eligibility_audit()` runs, then it reports that cell as device-ELIGIBLE — this
    is what makes the benchmark measure the new work rather than re-measuring the prior
    phase's.
- Invariants: no cell may be designed below the D-10-09 crossover `n` and then reported as
  a regression.
- Acceptance tests: a local dry-run mode (`--dry-run`) that enumerates the grid, prints
  the per-cell recipe and the eligibility audit, and exits WITHOUT touching a GPU — so the
  grid is reviewable before any Kaggle session is spent.
- Traceability: `bench/quick_gpu_speed/bench.py:38,220-225,287,315-336`;
  `bench/RESULTS.md` D-10-09; `bench/BENCH-03-SIGNOFF.md` (the correctness-gates-block-
  before-any-speed-number rigor template).
- Unresolved: whether official CatBoost GPU accepts every grid cell's recipe (e.g.
  `grow_policy="Depthwise"` under `task_type="GPU"`); PLAN's dry-run must confirm, and any
  unsupported cell is reported `N/A` and NEVER proxied (the `bench/RESULTS.md` Region
  precedent).

#### SPD-03 — Execute the grid on Kaggle P100 and publish the report
- Status: draft
- Principal failure reason: the phase claims "beats official CatBoost GPU" from a result
  whose variance straddles parity.
- Preconditions: SPD-01, SPD-02, and every Wave-1 spec merged (so the new axes are
  genuinely device-eligible).
- Output: `bench/full_param_gpu_speed/kaggle-output-<date>/{report.md,result.json}` plus a
  new dated run block appended to `bench/RESULTS.md` under its existing run-log template.
- Given/When/Then:
  - Given the grid completes, when the report is written, then EVERY cell reports
    `median(official) / median(catboost_rs)` with the min/max spread, and any cell whose
    spread crosses `1.0` is explicitly labelled "within noise", never claimed as a win.
  - Given the aggregate, when the headline claim is written, then it is qualified by the
    exact axes measured — never generalised beyond the grid.
- Acceptance tests: the report itself, reviewed against `bench/RESULTS.md`'s house style.
- Traceability: `bench/quick_gpu_speed/kaggle-output-260716-r4{a,b,c}/report.md` (the
  three prior runs whose 1.025×/1.141×, 0.988×/0.942×, 1.050×/1.051× spread is the reason
  this spec exists).
- Unresolved: Kaggle GPU session availability on account `yensen2` (already at the batch
  session cap this session) — a scheduling constraint, not a technical one.

## 6. Acceptance scenarios (summary)

| # | Scenario | Bar | Spec(s) | Wave |
|---|---|---|---|---|
| 1 | Default `boost_from_average=true` RMSE fit reaches device and matches upstream | ≤1e-5 | FPP-01,02,03,04 | 1 |
| 2 | `bias == 0.0` fits behave byte-unchanged | regression | FPP-01,02 | 1 |
| 3 | `leaf_estimation_method="Exact"` MAE and Quantile fits reach device, match upstream | ≤1e-5 | FPP-05,06,07,08 | 1 |
| 4 | Exact × LogCosh still declines (CPU-legal but device-uncovered) | correctness | FPP-06 | 1 |
| 5 | Combination/tensor CTR fit reaches device, structure AND leaf values match upstream | ≤1e-5 | FPP-09,10,11 | 1 |
| 6 | Simple-projection CTR fits behave byte-unchanged | regression | FPP-09 | 1 |
| 7 | Sampled Depthwise/Lossguide/Region fits reach device and match CPU | ε=1e-4 | FPP-12,13,14 | 1 |
| 8 | Region × Poisson still errors (no resident nonsym sampler) | correctness | FPP-13 | 1 |
| 9 | The seam signature change breaks nothing | regression | FPP-15 | 1 |
| 10 | `task_type="CPU"` accepted on any wheel, predictions bit-identical | correctness | FPP-16 | 2 |
| 11 | `task_type="GPU"` errors actionably on a cpu-only wheel | correctness | FPP-16 | 2 |
| 12 | Cross-gap composition: bias×weighted×CTR admits; exact×CTR, exact×sampling, combo-CTR×one-hot, multi-perm×combo-CTR, random_strength all still decline | correctness | FPP-17 | 5 |
| 13 | Ordered device fit matches upstream over all iterations | ≤1e-5 | FPP-18,19,20,21 | 4 |
| 14 | Buckets/BTMV/Counter CTR device columns bit-exact vs CPU; e2e ≤1e-5 | bit-exact / ≤1e-5 | FPP-22,23 | 4 |
| 15 | Monotone-constrained device fit matches upstream | ≤1e-5 | FPP-24,25 | 4 |
| 16 | Penalised device fit chooses the same splits as CPU | integer-equal | FPP-26,27 | 4 |
| 17 | Device fit with eval sets stops at the same iteration as CPU | correctness | FPP-28,29 | 4 |
| 18 | Pairwise / ranking / multi-output fits reach device and match upstream | ≤1e-5 | FPP-30..35 | 4 |
| 19 | Generator extension leaves the committed fixture bytes unchanged | regression | SPD-01 | 3 |
| 20 | The 16–24-cell grid runs in ONE Kaggle session with 3 repeats per cell | methodology | SPD-02,03 | 3 |
| 21 | The published claim is qualified by the exact axes measured, with variance | methodology | SPD-03 | 3 |

## 7. Impact scope

- **local**: `crates/cb-backend/src/gpu_runtime/session.rs` (resident approx seed, the
  nonsym/Region sample substitution, the five Phase-13 decline branches),
  `crates/cb-backend/src/kernels/` (any Wave-4 new kernel).
- **cross-module**: `crates/cb-train/src/boosting.rs` (`device_host_eligible`,
  `device_config` construction, `ctr_types_are_device_covered`,
  `build_device_ctr_config`) ↔ `crates/cb-backend` ↔ `crates/cb-compute`
  (`DeviceTrainConfig.bias`, `FamilyTreeArgs`, the `Runtime` trait signature).
- **external / public**: `crates/catboost-rs-py/src/params.rs` gains ONE newly-accepted
  Python parameter (`task_type`, D1). This is the only user-visible API change in the
  phase; everything else is internal device dispatch, observable only as "the same model,
  trained faster".
- **cross-service**: none (single Rust workspace).
- **operational**: one new `cargo test -p cb-train --no-default-features --features rocm
  --test <name>` invocation per new device e2e file must join the project's GPU
  verification checklist; ONE Kaggle P100 GPU session must be budgeted (D3).

## 8. Compatibility and migration

- No on-disk model format change. The device path must produce predictions numerically
  compatible with existing model files — only WHICH backend grows the tree changes, and
  (Track A) what the approximant starts from, which already matches the CPU path's own
  `starting_approx`.
- **One breaking-ish `Runtime` trait change** (FPP-15): `grow_tree_on_device` gains a
  parameter. `Runtime` is an internal trait with exactly one real override
  (`GpuBackend`); every other implementer inherits the default or wraps `GpuBackend`. No
  published API is affected.
- **One additive Python API change** (FPP-16): `task_type` moves from "rejected as a
  parity gap" to "accepted". This is strictly loosening — no previously-working call
  starts failing. `devices` / `device_config` remain `KnownNotYet`.
- No data migration, no config-file change, no dependency change.

## 9. Risks and open questions

- **Residual, documented, NOT closed by this phase**: the fixed-point histogram's
  `|Σ| < 2^33` overflow precondition (`crates/cb-backend/src/kernels.rs`) has no enforced
  runtime guard. Every new fixture must stay comfortably under
  `n · max(weight) · max(|der1|) < 2^33` by construction and document the margin at the
  generator call site, exactly as the prior phase's fixtures do. Carried from
  `.planning/plans/device-bootstrap-parity/SPEC.md` Finding F-D.
- **Residual (Track K)**: a 64-bit `fold_cat_hash` collision occurring on the CPU's
  string-hash fold but not on the device's bucket-code fold (or vice versa) would diverge
  the combined-projection partitions. Negligible at fixture scale, detected — not
  prevented — by FPP-11's ≤1e-5 bar. Documented, not guarded.
- **Scope risk (D2/D4)**: five spike tracks (Ordered, non-Borders CTR, monotone,
  penalties, plus three Phase-13 families) each carry an explicit escalation gate. The
  phase is genuinely maximal by the user's own directive; the gates are the mechanism that
  keeps "full parameter implementation" from becoming an unbounded overrun. If ANY gate
  fires, the correct response is to STOP that track and report — not to keep going.
- **Ordering discipline (applies to every gate relaxation)**: never flip a
  `device_host_eligible` clause before the underlying computation is verified correct.
  Every Track pairs its computation spec BEFORE its gate spec for exactly this reason
  (FPP-01→02, FPP-05→06, FPP-12→13, FPP-19→20, FPP-23, FPP-25, FPP-27, FPP-29).
- **Benchmark risk**: D3's budget assumes the whole grid fits one Kaggle session. SPD-02's
  `--dry-run` and running-elapsed-budget requirements exist to make an overrun a recorded,
  partial result rather than a lost session.
- **Unresolved implementation-detail items** (each flagged inline in its owning spec;
  listed here for visibility): FPP-07's admissible `border_count`; FPP-12's current
  `grow_one` dispatch line numbers; FPP-14's `replay_grow_draws` correctness for the
  non-symmetric level search; FPP-15's re-counted call-site list; FPP-16's
  `get_all_params()` interaction; FPP-21's re-evaluation of `ordered_boost_e2e/` after
  FPP-02 removes the bias clause; SPD-02's official-CatBoost cell support. None of these
  block this SPEC from being complete — they are exactly the class of detail `PLAN.md`
  resolves via CodeGraph (`SPEC.md` defines WHAT/WHY, `PLAN.md` defines the verified HOW).

## 10. Traceability and sources

- `[VERIFIED: LOCAL ./research.md]` — this phase's authoritative gap survey, itself
  grounded in `codegraph_explore` plus direct `Read`/`grep` verification (full source list
  in its own §Sources).
- `[VERIFIED: LOCAL ../gpu-device-coverage-expansion/{SPEC.md,PLAN.md}]` — the format,
  rigor and convention precedent this document follows (numbered locked decisions D1..Dn
  with rationale, typed contracts in §4, failure-isolated Given/When/Then specs in §5,
  Track structure, and the `PLAN.md` V-n / CORRECTIONS discipline).
- `[VERIFIED: LOCAL this session, base_commit 1535ac1]` — every line number, clause,
  signature, fixture recipe and grep result cited above was re-read from current `HEAD`
  during this planning pass, not carried from research.md. The full verification record,
  including four findings that CORRECT research.md, is `PLAN.md` §1 (V-1 … V-10).
- Locked decisions D1–D7: `[LOCAL, this session]` — made by the acting Planner/
  orchestrator from the user's verbatim directive plus this project's own `CLAUDE.md`
  constraints. Non-negotiable for this phase; conflicts are reported in `PLAN.md`
  §"CORRECTIONS to SPEC.md", never silently overridden.
