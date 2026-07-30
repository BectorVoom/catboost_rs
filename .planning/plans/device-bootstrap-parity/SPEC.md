---
title: "WR-01 — Device bootstrap parity (host-sampled Design A)"
status: draft
document_state: draft
format: markdown
spec_version: 1
updated_at: 2026-07-30T00:00:00Z
phase: device-bootstrap-parity
branch: fix/bootstrap-rng-draw-accounting
base_commit: 5a5068a
source_requirements:
  - "User: make `bootstrap_type != No` fits eligible for the device tree grower at the ≤1e-5 parity bar (WR-01, boosting.rs:3143-3159)."
  - "User decision 1: Design A (host RNG + host sampler + per-tree multiplier upload). Design B deferred."
  - "User decision 2: oracle route = regenerate bias-0 (`boost_from_average=False`) bootstrap fixtures."
  - "User decision 3: Bayesian/Bernoulli/MVS get real ≤1e-5 upstream parity; Poisson gets a separate, achievable, non-parity criterion."
  - "Research: .planning/plans/device-bootstrap-parity/research.md (authoritative, 18 HIGH findings)."
  - "Session probe: crates/cb-train/tests/device_oblivious_parity_probe_test.rs (uncommitted) on local gfx1151."
source_paths:
  - crates/cb-train/src/boosting.rs
  - crates/cb-train/src/bootstrap.rs
  - crates/cb-train/src/tree.rs
  - crates/cb-compute/src/runtime.rs
  - crates/cb-backend/src/gpu_backend.rs
  - crates/cb-backend/src/gpu_runtime/session.rs
  - crates/cb-backend/src/gpu_runtime/mod.rs
  - crates/cb-backend/src/kernels.rs
  - crates/cb-backend/src/kernels/bootstrap_device.rs
  - crates/cb-oracle/generator/gen_fixtures.py
treefinder_pending:
  reason: "TreeFinder MCP is configured but this repository's specification store is the plain-file `.planning/plans/<slug>/SPEC.md` corpus (12 sibling phases use it, no TreeFinder-indexed document exists for any of them). Registering this phase alone would fork the store. This file IS the draft spec of record; see §12."
  intended_identifier: "catboost-rs / .planning/plans/device-bootstrap-parity/SPEC.md"
tags: [wr-01, gpu, bootstrap, parity, rocm, cuda]
---

# WR-01 — Device bootstrap parity

> **Draft specification. NOT approved, accepted, final, or implemented.**
> Every specification below carries `implementation_state: unimplemented`.
>
> Evidence tags: `[VERIFIED: <path>:<line>]` — read from disk this session;
> `[VERIFIED: RUN]` — command output this session; `[RESEARCH]` — carried from
> `research.md` (already evidence-backed there); `[INFERRED]` — derived, stated as
> such; `[DECISION]` — a locked choice, not a finding.

---

## 1. Context

`train_inner` decides ONCE per fit whether the whole fit runs on the device
(`device_host_eligible`, `crates/cb-train/src/boosting.rs:3054-3098`). That gate
currently contains `&& matches!(params.bootstrap_type, EBootstrapType::No)`
`[VERIFIED: boosting.rs:3073]`, and the device config leaves `bootstrap_type`,
`sample_rate`, `mvs_lambda`, `rng_seed` at `DeviceTrainConfig::default()` behind
an explicit "WR-01 (NOT YET WIRED)" note `[VERIFIED: boosting.rs:3143-3159]`.

The device-side sampling apparatus already exists and is self-oracled
(`BootstrapState`, `MvsState`, `launch_bootstrap_weights_resident`,
`launch_mvs_weights_resident`, `fold_weights_resident`) `[RESEARCH]`, but wiring
it as-is would be numerically wrong on **both** stat channels, because
`grow_oblivious_tree_resident` feeds the SAME `(der1_h, weight_h)` pair into the
split histogram (`launch_partition_hist2_resident_into`, `mod.rs:3836-3852`) and
into the leaf-stat reduce (`launch_partition_update_into`, `mod.rs:3942-3951`),
whose channels 0/1 drive `cb_compute::calc_average` at `mod.rs:3963-3970`
`[VERIFIED: mod.rs:3836-3970]`. The session folds the sample into the SINGLE
`weight_ref` handed to that function `[VERIFIED: session.rs:1636-1645]`.

The CPU reference splits those two channels explicitly: the sample multiplies
`score_weighted_der1` and `score_weights` used for **split scoring only**
(`boosting.rs:3738-3770`), while leaf estimation uses the unsampled
`weighted_der1` / `weights` `[VERIFIED: boosting.rs:3723-3731]`.

### 1.1 What this session added to the research picture

Four findings that are **not** in `research.md` and that materially shape the plan:

- **F-A (probe result — Open Question 3 is ANSWERED YES).** The device oblivious
  grower already reproduces the CPU grower's *predictions* essentially exactly at
  `bootstrap_type = No`, bias 0, on the local gfx1151 rig:

  | shape | max abs Δpred | max abs Δleaf | split-mismatched trees |
  |---|---|---|---|
  | 512×4, depth 3, 5 iters | 2.22e-16 | 5.55e-17 | 0/5 |
  | 2048×8, depth 6, 10 iters | 2.22e-16 | 8.33e-17 | 0/10 |
  | 20000×16, depth 6, 20 iters | 4.44e-16 | 4.35e-1 | **3/20** |

  `[VERIFIED: RUN — user session, crates/cb-train/tests/device_oblivious_parity_probe_test.rs]`
  The ≤1e-5 bar is NOT blocked by the base grower. Research LOW-confidence item 23
  is retired.

- **F-B (NEW RISK — split ties).** At the largest shape the device and the CPU pick
  DIFFERENT splits on 3/20 trees while predictions still agree to 4e-16, i.e. the
  divergent splits are functionally degenerate *on that fixture*. The tie-break
  RULE is not the cause: the device folds per-block winners with
  `gain > best_gain || (gain == best_gain && cand < best_c)` over the flat
  candidate index `cand = feature * n_bins + bin` `[VERIFIED: mod.rs:3069-3078]`,
  which is lexicographic (feature ascending, then bin ascending); the CPU picks the
  strict-`>` first-wins border within a feature (`select_level_perturbed`,
  `tree.rs:1170-1180`) then the strict-`>` first-wins feature
  (`tree.rs:1189-1218`) — the SAME lexicographic order `[INFERRED from both
  sources]`. The cause is therefore **numeric**, see F-C.

- **F-C (mechanism — fixed-point histogram).** The device split histogram is a
  FIXED-POINT `Atomic<u64>` accumulator: every per-object contribution is
  quantized `round(v · 2^30) → i64 → u64` before the integer atomic add
  (`fixedpoint_encode`, `kernels.rs:3889-3893`, called at `kernels.rs:4025-4026`
  and `:4210-4211`; scale `REDUCE_FIXEDPOINT_SCALE_F64 = 2^30`,
  `kernels.rs:2335`) `[VERIFIED]`. The CPU histogram sums exact `f64`. So the two
  histograms are *deterministically different* by up to `2^-31 ≈ 4.66e-10` per
  object per channel, which is exactly the size of perturbation that flips a
  near-tie. This is a designed property (it buys run-to-run determinism, GPUT-06),
  not a defect — but it means **split-structure equality can never be an
  unconditional oracle assertion**, and it means bootstrap (which multiplies
  contributions and creates more equal-gain candidates) makes near-ties more
  frequent.

- **F-D (NEW RISK — fixed-point RANGE under sampling).** The same encode carries a
  documented precondition `|Σ| < 2^33` per accumulated bin, with **no in-kernel
  guard** — beyond it the wrapping `u64` add silently sign-flips
  `[VERIFIED: kernels.rs:2325-2335, 3884-3893]`. Sampling changes the magnitudes
  fed to that encode:
  - Bernoulli: `s ∈ {0, 1}` — strictly non-increasing, no new risk.
  - Bayesian: `s = (-FastLogf(u + 1e-100))^t`; with `u ≥ 2^-64`, `-ln u ≤ 44.4`, so
    `s ≤ 44.4^t` — bounded, modest at the fixture's `t = 1.0` `[INFERRED from
    bootstrap.rs:145-149 + TFastRng64::gen_rand_real1 range]`.
  - MVS: `s = 1/prob` with `prob = |der|_aug / threshold`, kept only when
    `prob > f64::EPSILON` `[VERIFIED: bootstrap.rs:315-327]`. The **der** channel is
    self-bounding (`|der · s| ≤ threshold · |der| / sqrt(λ + der²) ≤ threshold`),
    but the **weight** channel `w · s = threshold / sqrt(λ + der²)` is unbounded as
    `λ → 0` `[INFERRED — algebra on bootstrap.rs:307-323]`. MVS is therefore the
    one arm that can plausibly approach the `2^33` ceiling, and it needs an
    explicit host-side guard.

- **F-E (leaf reduce is NOT deterministic).** `partition_update_kernel` merges the
  leaf stats with a NAKED float atomic (`Atomic<F>::fetch_add`), explicitly
  documented as order-dependent, with the note that tree STRUCTURE is bit-identical
  run-to-run but LEAF VALUES / PREDICTIONS carry ulp-level run-to-run variance and
  "a strict bit-reproducibility claim for predictions is out of scope"
  `[VERIFIED: kernels.rs:3800-3826, 3852-3863]`. F-A measured that variance at
  ≤8.4e-17 in practice, but the ≤1e-5 sign-off must state it as a budget, not
  assume it away.

- **F-F (Open Question 1 is RESOLVED).** `catboost 1.2.10` is importable in this
  environment (system Python 3.14.6 and `.venv`) `[VERIFIED: RUN
  `python3 -c "import catboost"` → `1.2.10`]`. The bias-0 fixture route (user
  decision 2) is executable here; the generator already has a
  `--<wave>-only` targeted-entrypoint convention to avoid overwriting committed
  fixtures `[VERIFIED: gen_fixtures.py:3246-3272]`.

### 1.2 Locked decisions carried into this spec

- **[DECISION D1] Design A.** Host keeps `bootstrap()` and the whole `5a5068a` RNG
  draw model. Per tree the host uploads one `n`-length `f64` multiplier
  `s[i] = control[i] ? sample_weights[i] : 0.0`. No new CubeCL kernel:
  `fold_weights_resident` is a generic elementwise `f64` product over
  `vector_mul_kernel` `[VERIFIED: bootstrap_device.rs:417-439]`, reused for BOTH
  `der1 · s` and `weight · s`.
- **[DECISION D2] Oracle route = regenerate bias-0 fixtures** into a NEW
  `bootstrap_dev/` family; the committed `bootstrap/` family
  (`boost_from_average=True`) stays byte-untouched and must stay green.
- **[DECISION D3] Type scope.** Bayesian / Bernoulli / MVS → real ≤1e-5
  device-vs-upstream. Poisson → §8, a distinct non-parity contract.
- **[DECISION D4] Explicit intent flag.** `DeviceTrainConfig` gains
  `sample_from_host: bool` rather than implying host sampling from "config says
  `No` but a sample arrived" (research §A1 alternative, recommended there). This
  makes double-sampling (host multiplier AND device sampler) structurally
  impossible instead of merely unlikely.

---

## 2. Scope

### In scope

- Device score/leaf channel split in `grow_oblivious_tree_resident` + `grow_one`.
- Seam widening `grow_tree_on_device(approx, target, sample)` across its 3 impls
  `[VERIFIED: RUN — `grep -rn "fn grow_tree_on_device"` → `runtime.rs:1279`,
  `gpu_backend.rs:310`, `tests/device_seam_test.rs:69`]`.
- Host per-tree sample construction + RNG draw replay + `prev_leaf_mean_l2` carry
  on the device branch.
- `device_host_eligible` relaxation for `{Bayesian, Bernoulli, Mvs}` + config
  threading.
- A fixed-point RANGE precondition guard for the sampled score channels (F-D).
- New bias-0 upstream fixture family + its CPU oracle and its device oracle.
- Promotion of the session probe into a committed base-grower oracle, and a
  split tie-break characterisation with a locked, documented rule.
- Poisson behavioural contract (§8).

### Out of scope (explicitly, not silently)

- `random_strength != 0` on device (no perturbation arm) — gate keeps
  `params.random_strength == 0.0` `[VERIFIED: boosting.rs:3074]`.
- Weighted pools (WR-03, `Σ w·der` gap) — gate keeps `weights.all(== 1.0)`
  `[VERIFIED: boosting.rs:3086]`.
- Lifting CR-01 (`bias == 0.0`, `boosting.rs:3092`) — D2 makes it unnecessary.
- Non-symmetric / Region / CTR / exact-leaf × bootstrap combinations.
- Design B / B′ device-resident sampling (perf follow-up).
- Converting `partition_update_kernel` to fixed-point atomics — characterised here
  (WR01-S13), scheduled only if its determinism gate fails.
- Ordered boosting, multi-dimensional approx.

---

## 3. Dependency context (behaviourally relevant only)

| Dependency | Why it constrains this phase |
|---|---|
| `cb_train::bootstrap::bootstrap` (`bootstrap.rs:383-439`) | the ≤1e-5-verified CPU sampler and the sole source of `sample_weights`/`control`; bit-sensitive (`fast_log2f`), never re-derive |
| `cb_train::bootstrap::last_iter_mean_leaf_value` (`bootstrap.rs:363-369`) | per-tree MVS λ input; the device branch must carry it |
| `cb_core::TFastRng64` + `cb_core::std_normal` | the draw model; `std_normal`'s `gen_rand` count is data-dependent (Marsaglia polar) so the replay must CALL it, never count it |
| `PRE_TREE_DRAWS = 2` (`boosting.rs:58`), `POST_TREE_EXTRA_DRAWS = 2` (`boosting.rs:68`) | instrumented-ground-truth constants from `5a5068a`; reuse, do not re-derive |
| per-level draws in `greedy_tensor_search_oblivious_perturbed` (`tree.rs:610-614`) + `select_level_perturbed` (`tree.rs:1142`, `:1200`, `:1215`) | the exact replay shape: `n_features × gen_rand_real1`, then `1 × gen_rand`, then `n_features × std_normal` (border-less features included) |
| `fold_weights_resident` (`bootstrap_device.rs:417-439`) | the ONLY multiply primitive needed; no new kernel |
| `fixedpoint_encode` / `REDUCE_FIXEDPOINT_SCALE_F64 = 2^30` (`kernels.rs:2335`, `:3889`) | quantizes the histogram; source of F-C and the F-D range precondition |
| `partition_update_kernel` naked float atomic (`kernels.rs:3826-3862`) | source of F-E prediction-level run-to-run variance |
| `cb_compute::calc_average` / `scale_l2_reg` | frozen leaf/L2 math; `scaled_l2` is tree-invariant and sampling-independent `[RESEARCH finding 9]` |
| `DeviceTrainConfig` (`runtime.rs:1082-1141`) | plain-host only (no `cubecl` type may appear); additive fields are safe (`..default()` at call sites) |
| `cb-backend` must NEVER depend on `cb-train` | Design A satisfies this trivially (sampler stays host-side) |
| `crates/cb-oracle/generator/gen_fixtures.py` + `catboost==1.2.10` | the only source of a true upstream oracle; `--*-only` entrypoint convention protects committed fixtures |

---

## 4. Type contract

### 4.1 Proposed / changed types

```rust
// crates/cb-compute/src/runtime.rs — ADDITIVE field on the existing plain-host struct.
pub struct DeviceTrainConfig {
    // ... existing fields unchanged ...
    /// PROPOSED (WR-01, D4). `true` iff the HOST computes the per-tree bootstrap
    /// sample and passes it through `Runtime::grow_tree_on_device`'s `sample`
    /// argument. Mutually exclusive with the device-resident samplers: when
    /// `true`, `bootstrap_type` describes WHICH host sampler ran (for gate
    /// bookkeeping / diagnostics) and the session MUST NOT open `BootstrapState`
    /// or `MvsState`.
    pub sample_from_host: bool,
}
// Default: `false` (byte-unchanged covered regime).
```

```rust
// crates/cb-compute/src/runtime.rs — seam widening (3 impl sites + 1 caller).
fn grow_tree_on_device(
    &self,
    approx: &[f64],   // len n, dimension-major (dim == 1 here)
    target: &[f64],   // len n
    sample: &[f64],   // len 0 (no host sampling) OR len n (per-object multiplier)
) -> CbResult<Option<DeviceGrownTree>>;
```

```rust
// crates/cb-backend/src/gpu_runtime/mod.rs — four-channel grow (existing fn, widened).
fn grow_oblivious_tree_resident(
    client: &ComputeClient<SelectedRuntime>,
    approx_h: Handle,
    der1_h: &Handle,          // UNSAMPLED — leaf estimation + routing + next-tree der
    weight_h: &Handle,        // UNSAMPLED — leaf estimation
    score_der1_h: &Handle,    // PROPOSED: der1 · s  — split histogram ONLY
    score_weight_h: &Handle,  // PROPOSED: weight · s — split histogram ONLY
    /* ... existing args unchanged ... */
) -> CbResult<(GrownTree, Handle, Handle)>;
```

```rust
// crates/cb-train/src/device_draw_replay.rs — PROPOSED new module (host-side, pure).
/// Replay, on `rng`, exactly the draws the CPU oblivious grow would have consumed
/// for one tree of `depth` levels over `n_features` listed float features.
pub fn replay_grow_draws(rng: &mut TFastRng64, depth: usize, n_features: usize);
```

### 4.2 Existing types reused unchanged

`BootstrapResult { sample_weights: Vec<f64>, control: Vec<bool> }`
(`bootstrap.rs:87-93`), `EBootstrapType`, `DeviceBootstrapType`
(`runtime.rs:1020-1032`), `DeviceGrownTree`, `BestSplit`, `CbError` / `CbResult`.

---

## 5. Specifications

Each specification is sized so one focused test demonstrates it and a failure has
one principal cause.

---

### SPEC WR01-S1: The device split histogram consumes the SAMPLED stat pair

- **Implementation state:** unimplemented
- **Document state:** draft
- **Principal failure reason:** the split histogram is filled from a stat pair
  other than `(der1·s, weight·s)`.
- **Scope:** `crates/cb-backend/src/gpu_runtime/mod.rs::grow_oblivious_tree_resident`,
  the `launch_partition_hist2_resident_into` call at `:3836-3852`.
- **Dependencies:** `launch_partition_hist2_resident_into` — the fill; the
  subtraction-trick parent chain at `:3861-3869` — stays valid only if the SAME
  pair is used at every level.
- **Input type:** `score_der1_h: &Handle`, `score_weight_h: &Handle` (each `n` f64
  device elements).
- **Output type:** unchanged `CbResult<(GrownTree, Handle, Handle)>`.
- **Preconditions:** `score_*_h` are length `n`; when no host sampling is active
  they are the same handles as `der1_h` / `weight_h`.
- **Behavior:** every level's fill (level 0 and the filtered levels > 0) reads
  `score_der1_h` / `score_weight_h`; the derive-sibling subtraction operates on
  histograms built from that same pair.
- **Postconditions:** the chosen `(feature, bin)` per level is a function of the
  sampled stats only.
- **Errors:** unchanged (`CbError::OutOfRange` on index/overflow guards,
  `CbError::Degenerate` on a no-candidate level).
- **Side effects:** none beyond existing device allocations.
- **Acceptance examples:**
  - Given `s ≡ 1.0`, when the tree is grown, then the split sequence is identical
    to the `s`-empty grow (bit-for-bit).
  - Given a hand-built `s` with a subset zeroed, when the tree is grown, then the
    level-0 chosen split equals a host-computed reference argmax over the same
    masked histogram.
- **Evidence:** `mod.rs:3826-3931` (level loop), `:3836-3852` (fill), `:3861-3869`
  (derive), `:3877-3879` (score).
- **Non-goals:** changing the score function, the padding rule, or the tie-break.

---

### SPEC WR01-S2: The device leaf estimate consumes the UNSAMPLED stat pair

- **Implementation state:** unimplemented
- **Document state:** draft
- **Principal failure reason:** leaf values are computed from sampled stats
  (currently both channels are wrong under sampling — the latent bug F/finding 10).
- **Scope:** `grow_oblivious_tree_resident`, `launch_partition_update_into` at
  `mod.rs:3942-3951` and the `calc_average` loop at `:3963-3970`.
- **Dependencies:** `cb_compute::calc_average` (frozen); `scaled_l2` (tree-invariant,
  sampling-independent); `launch_apply_leaf_delta_into` at `:3974`;
  `launch_der_binary_resident` at `:3978` (next tree's der1 — MUST stay on the
  unsampled approx).
- **Input type:** `der1_h: &Handle`, `weight_h: &Handle` (unsampled, length `n`).
- **Output type:** `GrownTree.leaf_values: Vec<f64>` (UNSCALED — the
  `DeviceGrownTree` contract; `cb-train` applies `learning_rate`).
- **Preconditions:** `leaf_of_h` is final after the level loop.
- **Behavior:** `part_stats` channels 0/1 are `Σ der1` / `Σ weight` over the FULL
  object set; `leaf_values[l] = calc_average(part_stats[3l], part_stats[3l+1],
  scaled_l2)`.
- **Postconditions:** for a fit with sampling active, every device leaf value
  equals the CPU leaf value for the same tree structure to within the F-E budget.
- **Errors:** unchanged.
- **Side effects:** the resident approx and der1 advance from the UNSAMPLED leaves.
- **Acceptance examples:**
  - Given a hand-built `s` that zeroes half the objects, when one tree is grown,
    then `Σ_leaves part_stats[3l+1]` equals `n` (not the kept count).
  - Given the same `s`, when the leaf values are read, then they equal a host
    reference `calc_average` over the UNSAMPLED per-leaf sums.
- **Evidence:** `mod.rs:3942-3970`; CPU reference `boosting.rs:3723-3731`.
- **Non-goals:** the Newton / exact-leaf arms.

---

### SPEC WR01-S3: An empty sample is byte-identical to today's grow (D-04)

- **Implementation state:** unimplemented
- **Document state:** draft
- **Principal failure reason:** the refactor perturbs the `bootstrap_type = No`
  device path.
- **Scope:** `grow_oblivious_tree_resident`, `GpuTrainSession::grow_one`,
  `GpuBackend::grow_tree_on_device`.
- **Dependencies:** every shipped device test
  (`device_nonsym_fit_test`, `device_region_fit_test`,
  `session_depth_gt1_test`, `kernels::grow_loop`).
- **Input type:** `sample: &[f64]` with `sample.is_empty() == true`.
- **Output type:** `CbResult<Option<DeviceGrownTree>>`.
- **Preconditions:** the fit did not request host sampling.
- **Behavior:** no upload, no fold launch; `score_*_h` alias `der1_h` / `weight_h`;
  the emitted tree is bit-identical to the pre-change path.
- **Postconditions:** all existing device oracles stay green with unchanged values.
- **Errors:** none added.
- **Side effects:** none.
- **Acceptance examples:**
  - Given a fixed fixture at `bootstrap_type = No`, when grown before and after the
    change, then splits, `leaf_of`, and `part_stats` are equal element-for-element.
- **Evidence:** current single-pair call sites `mod.rs:3836-3852`, `:3942-3951`;
  current session pass-through `session.rs:1636-1645`.
- **Non-goals:** performance.

---

### SPEC WR01-S4: The per-tree sample crosses the seam, length-validated

- **Implementation state:** unimplemented
- **Document state:** draft
- **Principal failure reason:** a wrong-length or silently-ignored sample reaches
  the grow.
- **Scope:** `cb_compute::Runtime::grow_tree_on_device` (trait default + doc),
  `GpuBackend::grow_tree_on_device` (`gpu_backend.rs:310-341`),
  `GpuTrainSession::grow_one` upload, `FakeRuntime`
  (`tests/device_seam_test.rs:69`).
- **Dependencies:** `upload_channel_floats` (`mod.rs:1709`) — the existing upload
  helper; the existing `approx` length check at `gpu_backend.rs:321-327` — the
  pattern to mirror.
- **Input type:** `sample: &[f64]` (len `0` or `n`).
- **Output type:** `CbResult<Option<DeviceGrownTree>>`.
- **Preconditions:** the session is open.
- **Behavior:** `sample.len() ∈ {0, n}`; any other length is rejected BEFORE any
  launch; the trait default binds `sample` and returns `Ok(None)`.
- **Postconditions:** a non-empty sample produces exactly two extra
  `fold_weights_resident` launches per tree and one `n`-element upload.
- **Errors:** `CbError::LengthMismatch { column: "sample", expected: n, actual }`.
- **Side effects:** one `8·n`-byte host→device transfer per tree.
- **Acceptance examples:**
  - Given `sample.len() == n - 1`, when `grow_tree_on_device` is called, then it
    returns `Err(CbError::LengthMismatch { column: "sample", .. })` and no kernel
    is launched.
  - Given a `Runtime` impl that does not override the seam, when called with a
    non-empty sample, then it returns `Ok(None)`.
- **Evidence:** `runtime.rs:1279-1286`; `gpu_backend.rs:310-341`.
- **Non-goals:** zero-copy / pinned-memory upload.

---

### SPEC WR01-S5: Host sampling and device-resident sampling are mutually exclusive

- **Implementation state:** unimplemented
- **Document state:** draft
- **Principal failure reason:** both samplers run, multiplying the sample twice.
- **Scope:** `DeviceTrainConfig::sample_from_host` (new),
  `DeviceTrainConfig::is_covered_regime`, the `None` gate arm at
  `session.rs:1038-1092`, `GpuTrainSession` field init.
- **Dependencies:** `BootstrapState` / `MvsState` construction; the existing
  `bootstrap_covered` / `mvs_covered` booleans (`session.rs:1058-1074`).
- **Input type:** `&DeviceTrainConfig`.
- **Output type:** `CbResult<Option<GpuTrainSession>>` (open / decline).
- **Preconditions:** none.
- **Behavior:** when `sample_from_host == true` the session opens the oblivious
  covered regime and leaves `self.bootstrap` and `self.mvs` as `None`, regardless
  of `bootstrap_type` / `mvs_lambda`; when `sample_from_host == false` behaviour is
  byte-unchanged.
- **Postconditions:** at most one of {host multiplier, device sampler} is ever
  applied in a fit.
- **Errors:** decline (`Ok(None)`) for any other non-default family flag
  (D-10-01 all-or-nothing).
- **Side effects:** none.
- **Acceptance examples:**
  - Given `sample_from_host = true, bootstrap_type = Bayesian`, when `begin` runs,
    then the session opens AND `grow_one` consumes zero draws from the session's own
    `TFastRng64`.
  - Given `sample_from_host = false, bootstrap_type = Bayesian` (the pre-existing
    device-resident regime), when `begin` runs, then behaviour is unchanged.
- **Evidence:** `session.rs:1038-1092`, `:1546-1612`; `runtime.rs:1127-1141`.
- **Non-goals:** removing the device samplers (retained as the Design B asset).

---

### SPEC WR01-S6: The host builds the per-tree sample multiplier from `bootstrap()`

- **Implementation state:** unimplemented
- **Document state:** draft
- **Principal failure reason:** the multiplier is not
  `control[i] ? sample_weights[i] : 0.0`.
- **Scope:** the device branch in `train_inner` (`boosting.rs:3196-3486`).
- **Dependencies:** `bootstrap()`; `weighted_der1` / `der_obj` construction
  (`boosting.rs:3634-3695`) — MVS is the only arm reading derivatives
  (`bootstrap.rs:410-412`); `runtime.compute_gradients` (to produce `der_obj` on
  the device branch, where the host owns `approx`).
- **Input type:** `EBootstrapType`, `&[f64]` der (MVS only), `f64` subsample,
  `f32` bagging_temperature, `Option<f64>` prev λ input, `&mut TFastRng64`.
- **Output type:** `Vec<f64>` of length `n`.
- **Preconditions:** unit weights (gate), `approx_dimension == 1` (gate).
- **Behavior:** `s[i] = if control[i] { sample_weights[i] } else { 0.0 }` — the
  exact CPU `score_weights` at unit weight and the exact `score_weighted_der1`
  multiplier; MVS computes `der_obj` first, Bernoulli/Bayesian pass a length-`n`
  placeholder without an extra gradient round-trip.
- **Postconditions:** `s.len() == n`; `s` is finite; `s[i] >= 0.0`.
- **Errors:** propagates `bootstrap()`'s `CbResult`.
- **Side effects:** advances the persistent host RNG by exactly the draws
  `bootstrap()` consumes.
- **Acceptance examples:**
  - Given Bernoulli `subsample = 0.8` and a fixed seed, when `s` is built, then
    `s[i] ∈ {0.0, 1.0}` and its non-zero count equals the CPU `control` true count
    for the same seed.
  - Given MVS, when `s` is built, then it equals the CPU `score_weights` vector for
    the same tree, element-for-element.
- **Evidence:** `boosting.rs:3699-3770`; `bootstrap.rs:383-439`.
- **Non-goals:** multi-dimensional approx.

---

### SPEC WR01-S7: The device branch replays the CPU grow's RNG draws exactly

- **Implementation state:** unimplemented
- **Document state:** draft
- **Principal failure reason:** the RNG stream position after a device tree differs
  from after a CPU tree, so tree ≥ 1 samples differ.
- **Scope:** new `crates/cb-train/src/device_draw_replay.rs::replay_grow_draws`,
  called from the device branch.
- **Dependencies:** `tree.rs:610-614` (per-level RSM: `n_features ×
  gen_rand_real1`), `tree.rs:1142` (1 × `gen_rand`), `tree.rs:1189-1218` (one
  `std_normal` per LISTED feature including border-less ones),
  `PRE_TREE_DRAWS`, `POST_TREE_EXTRA_DRAWS`, `cb_core::std_normal`.
- **Input type:** `(&mut TFastRng64, depth: usize, n_features: usize)`.
- **Output type:** `()` (the RNG is advanced in place).
- **Preconditions:** `draws_active == true`.
- **Behavior:** per level, in order: `n_features` `gen_rand_real1`, then one
  `gen_rand`, then `n_features` literal `std_normal` calls. The device branch wraps
  this with `PRE_TREE_DRAWS` `gen_rand` before `bootstrap()` and
  `POST_TREE_EXTRA_DRAWS` `gen_rand` after the fold.
- **Postconditions:** for identical `(seed, depth, n_features)` the RNG's
  `raw_state()` after `replay_grow_draws` equals its state after a real
  `greedy_tensor_search_oblivious_perturbed` grow with `score_st_dev = 0.0`.
- **Errors:** none.
- **Side effects:** RNG advance only.
- **Acceptance examples:**
  - Given seed `S`, depth `3`, a 4-feature matrix (one feature border-less), when
    the real perturbed grow runs on one clone and `replay_grow_draws` on another,
    then both RNGs report the same `call_count()` AND the same `raw_state()`.
  - Given the same setup, when `std_normal` is replaced by a fixed-count formula,
    then the test FAILS (guard against the data-dependent-rejection trap).
- **Evidence:** `tree.rs:596-649`, `:1141-1218`; `boosting.rs:58,68,3665-3671,4580-4584`;
  GROUND_TRUTH.md:60-71 (observed `std_normal` `gen_rand` counts 8/10/12/14 at
  `n_features = 4`).
- **Non-goals:** replaying non-symmetric / Region / CTR grow draws.

---

### SPEC WR01-S8: The device branch carries `prev_leaf_mean_l2` for MVS λ

- **Implementation state:** unimplemented
- **Document state:** draft
- **Principal failure reason:** MVS λ is frozen across trees, so every tree ≥ 1
  samples from the wrong distribution.
- **Scope:** the device branch's oblivious fold arm (`boosting.rs:3319-3400`),
  mirroring `boosting.rs:4586-4589`.
- **Dependencies:** `last_iter_mean_leaf_value`; `normalize_leaf_values` (the
  device leaf values are `learning_rate`-scaled at `:3376-3383`, matching the CPU
  arm which feeds the SCALED `leaf_values` into `last_iter_mean_leaf_value`).
- **Input type:** `&[f64]` (this tree's stored, lr-scaled leaf values).
- **Output type:** `Option<f64>` assigned to `prev_leaf_mean_l2`.
- **Preconditions:** the oblivious device arm produced leaf values.
- **Behavior:** after the fold and before `continue`,
  `prev_leaf_mean_l2 = Some(last_iter_mean_leaf_value(&device_leaf_values))`.
- **Postconditions:** tree `k ≥ 1`'s MVS λ equals the CPU's for the same model.
- **Errors:** none.
- **Side effects:** none.
- **Acceptance examples:**
  - Given an MVS fit of ≥ 3 iterations, when the device λ inputs are logged against
    the CPU λ inputs, then they agree for every tree.
  - Given the carry is removed, then the MVS device-vs-CPU oracle fails from tree 1.
- **Evidence:** `boosting.rs:3486` (today: `continue` with no carry),
  `:4586-4589` (the CPU carry), `bootstrap.rs:344-369`.
- **Non-goals:** the non-symmetric / Region device arms (out of scope for
  bootstrap this phase).

---

### SPEC WR01-S9: The eligibility gate admits Bayesian / Bernoulli / MVS

- **Implementation state:** unimplemented
- **Document state:** draft
- **Principal failure reason:** the wrong set of bootstrap types reaches (or fails
  to reach) the device.
- **Scope:** `device_host_eligible` (`boosting.rs:3054-3098`) and `device_config`
  (`boosting.rs:3133-3160`, including replacing the stale WR-01 note).
- **Dependencies:** every other gate clause stays unchanged (`random_strength == 0`,
  unit weights, `bias == 0`, Gradient/Simple leaf, SymmetricTree for this phase).
- **Input type:** `&BoostParams`.
- **Output type:** `bool` + a `DeviceTrainConfig`.
- **Preconditions:** none.
- **Behavior:** the bootstrap clause becomes
  `matches!(params.bootstrap_type, EBootstrapType::No | Bayesian | Bernoulli | Mvs)`
  — an explicit allow-list, NOT `!= Poisson`, so a future `EBootstrapType` variant
  cannot silently become device-eligible. The config threads
  `sample_from_host = bootstrap_type != No`, the mapped `bootstrap_type`, and
  `sample_rate = params.subsample as f32`; `mvs_lambda` stays `None` (Design A does
  not use the device sampler) and `rng_seed` stays `0` (the host owns the stream).
- **Postconditions:** a Bernoulli/Bayesian/MVS fit that satisfies every other clause
  takes the device path; a Poisson fit never does (§8).
- **Errors:** none (a declined fit falls to the CPU path, D-04).
- **Side effects:** more real fits now reach the device via
  `crates/catboost-rs/src/builder.rs:378-392`.
- **Acceptance examples:**
  - Given Bernoulli + every other clause satisfied, when `train` runs on a device
    build, then the fit is device-grown (observable device signature asserted).
  - Given Bernoulli + `random_strength = 0.5`, when `train` runs, then the fit is
    CPU-grown.
  - Given a hypothetical new `EBootstrapType` variant, when added, then the gate
    does not compile until the allow-list is updated (exhaustive `matches!`).
- **Evidence:** `boosting.rs:3054-3098`, `:3133-3160`.
- **Non-goals:** relaxing any other clause.

---

### SPEC WR01-S10: The sampled score channels satisfy the fixed-point range precondition

- **Implementation state:** unimplemented
- **Document state:** draft
- **Principal failure reason:** a sampled contribution overflows the `2^30`
  fixed-point encode and the histogram silently sign-flips (F-D).
- **Scope:** a host-side pre-launch guard in `GpuTrainSession::grow_one` (or the
  `cb-train` device branch — see the task file's placement note), covering the
  channels handed to `launch_partition_hist2_resident_into`.
- **Dependencies:** `REDUCE_FIXEDPOINT_SCALE_F64 = 2^30` (`kernels.rs:2335`); the
  documented precondition `|Σ_bin| < 2^33` (`kernels.rs:2325-2334`); MVS's
  unbounded weight channel `w·s = threshold / sqrt(λ + der²)`.
- **Input type:** the host `sample: &[f64]` plus the session's `n` and the host's
  knowledge of `max|der1|` (already available on the device branch, which owns
  `approx`).
- **Output type:** `CbResult<()>` (guard) — `Ok(())` on the safe path.
- **Preconditions:** host sampling active.
- **Behavior:** reject BEFORE launch when the conservative bound
  `n · max|der1·s| ≥ 2^33` or `n · max|w·s| ≥ 2^33`; also reject a non-finite or
  negative `s[i]`.
- **Postconditions:** a fit that would silently produce a sign-flipped histogram
  errors instead.
- **Errors:** `CbError::OutOfRange` naming the channel, the offending bound, and
  the `2^33` limit.
- **Side effects:** an O(n) host scan per tree (negligible next to the O(n) upload).
- **Acceptance examples:**
  - Given a synthetic `s` with one element `1e12` and `n = 1000`, when the guard
    runs, then it returns `Err(CbError::OutOfRange)` and no kernel is launched.
  - Given the fixtures of this phase, when the guard runs, then it returns `Ok(())`
    on every tree (so the guard never fires in normal operation).
  - Given `s[i] = f64::NAN`, then `Err(CbError::OutOfRange)`.
- **Evidence:** `kernels.rs:2320-2335`, `:3884-3893`; `bootstrap.rs:307-327`.
- **Non-goals:** changing the fixed-point scale or adding an in-kernel guard
  (a `#[cube]` kernel cannot surface a typed error).

---

### SPEC WR01-S11: The base device oblivious grower holds ≤1e-5 vs the CPU grower

- **Implementation state:** unimplemented
- **Document state:** draft
- **Principal failure reason:** the `bootstrap_type = No` device grower cannot meet
  the phase's ≤1e-5 bar, making every downstream bootstrap claim unreachable.
- **Scope:** a committed integration oracle promoted from the session probe
  (`crates/cb-train/tests/device_oblivious_parity_test.rs`, replacing the
  uncommitted `*_probe_test.rs`).
- **Dependencies:** `GpuBackend`; a local `CpuRefRuntime` that declines the seam
  (the `device_nonsym_fit_test.rs:106-126` pattern — `CpuBackend` is not compiled
  under rocm/cuda); `cb_model::predict_raw`.
- **Input type:** three fixture shapes (512×4 d3 ×5, 2048×8 d6 ×10,
  20000×16 d6 ×20), `bootstrap_type = No`, bias 0, unit weights,
  `random_strength = 0`.
- **Output type:** pass/fail plus a printed `max|Δpred|`, `max|Δleaf|`, and
  split-mismatch count per shape.
- **Preconditions:** built with `--no-default-features --features rocm` (or `cuda`);
  otherwise the test SKIPs with an explicit message.
- **Behavior:** asserts `max|Δpred| ≤ 1e-5` (the phase bar, NOT the shipped ε=1e-4);
  REPORTS split mismatches without asserting equality (see WR01-S12 for why
  structure equality is not assertable).
- **Postconditions:** the ≤1e-5 claim for the base path is a committed, running
  gate rather than a one-off measurement.
- **Errors:** none.
- **Side effects:** none.
- **Acceptance examples:**
  - Given the 20000×16 shape on gfx1151, when the test runs, then `max|Δpred|` is
    ≤1e-5 (measured 4.44e-16) and the split-mismatch count is printed (measured
    3/20).
  - Given a cpu/wgpu build, when the test runs, then it prints `SKIP …` and passes
    without touching a device.
- **Evidence:** F-A measurements; `device_nonsym_fit_test.rs:92-210` (pattern);
  ε=1e-4 precedents at `device_nonsym_fit_test.rs:167`,
  `bootstrap_device_test.rs:25`, `mvs_device_test.rs:26` (the creep this spec
  refuses).
- **Non-goals:** asserting split-structure equality; benchmarking.

---

### SPEC WR01-S12: Device/CPU split selection has one documented, locked tie-break order

- **Implementation state:** unimplemented
- **Document state:** draft
- **Principal failure reason:** a gain tie is resolved differently on device and
  CPU on a NON-degenerate split, producing a genuinely different model.
- **Scope:** a characterisation test
  (`crates/cb-train/tests/device_split_tiebreak_test.rs`) plus a documented
  invariant recorded on `score_partition_over_binsums` and `select_level_perturbed`.
- **Dependencies:** device fold rule `gain > best || (gain == best && cand < best_c)`
  over `cand = feature*n_bins + bin` (`mod.rs:3069-3078`) and the in-kernel
  per-block argmin it feeds; CPU per-border strict-`>` first-wins
  (`tree.rs:1175-1179`) then per-feature strict-`>` first-wins
  (`tree.rs:1201-1204`); the fixed-point histogram (F-C).
- **Input type:** (a) a synthetic fixture engineered to produce EXACT gain ties
  across two features and across two borders of one feature; (b) the 20000×16
  shape that empirically mismatches.
- **Output type:** assertions + a written characterisation.
- **Preconditions:** device build for the device half; the CPU half runs anywhere.
- **Behavior:**
  1. On an exact-tie fixture the device and the CPU MUST choose the same
     `(feature, bin)` — the lexicographic-lowest candidate. This is the RULE claim.
  2. On the empirically-mismatching shape the test attributes each mismatch: it
     recomputes both candidates' CPU gains and asserts the gain gap is below the
     fixed-point resolution floor `n · 2^-31` — i.e. the mismatch is explained by
     F-C quantization and not by a rule difference. A mismatch with a gain gap
     ABOVE that floor is a hard failure.
  3. The prediction-level consequence is bounded by the WR01-S11 / WR01-S14
     tolerance; split-structure equality is explicitly NOT an oracle assertion, and
     the reason is recorded in the test's module doc.
- **Postconditions:** any future divergence is diagnosable as rule-vs-numeric in one
  test run.
- **Errors:** none.
- **Side effects:** none.
- **Acceptance examples:**
  - Given two features whose best candidates have bit-identical gains, when both
    growers run, then both pick the lower feature index.
  - Given a mismatching tree at 20000×16, when the two candidates' CPU gains are
    compared, then `|gain_dev_choice − gain_cpu_choice| < n · 2^-31`.
  - Given an artificially inflated gain gap, then the test fails (it is not
    vacuous).
- **Evidence:** `mod.rs:3040-3088`; `tree.rs:1149-1223`; `kernels.rs:2325-2335`,
  `:3889-3893`; F-B.
- **Non-goals:** eliminating the mismatches (that would require abandoning the
  fixed-point histogram, i.e. abandoning GPUT-06 determinism).

---

### SPEC WR01-S13: The device leaf reduce's nondeterminism stays inside the ≤1e-5 budget

- **Implementation state:** unimplemented
- **Document state:** draft
- **Principal failure reason:** repeated identical device fits produce predictions
  that differ by more than the sign-off tolerance (F-E).
- **Scope:** a determinism oracle over `train()` on a device build, exercising the
  sampled path at the largest in-phase shape.
- **Dependencies:** `partition_update_kernel`'s naked float atomic
  (`kernels.rs:3826-3862`); the documented scope note at `kernels.rs:3800-3812`.
- **Input type:** the same fit executed `k = 5` times in one process.
- **Output type:** `max|Δpred|` across runs.
- **Preconditions:** device build.
- **Behavior:** tree STRUCTURE (splits) must be bit-identical across runs (the
  fixed-point histogram guarantees it); predictions must agree to ≤1e-7 — a full two
  decades inside the ≤1e-5 sign-off bar, so leaf-reduce jitter can never be the
  thing that breaks parity.
- **Postconditions:** the ≤1e-5 upstream claim is not silently dependent on a lucky
  atomic schedule.
- **Errors:** none.
- **Side effects:** none.
- **Acceptance examples:**
  - Given 5 identical Bayesian device fits, when compared pairwise, then split
    sequences are equal and `max|Δpred| ≤ 1e-7`.
  - Given the budget is exceeded, then the task file's documented escalation applies
    (promote `partition_update_kernel` to fixed-point `Atomic<u64>`, mirroring the
    histogram fill — the fix the kernel doc itself names).
- **Evidence:** `kernels.rs:3800-3826`; F-A leaf deltas 5.55e-17 / 8.33e-17.
- **Non-goals:** proving cross-hardware bit-identity.

---

### SPEC WR01-S14: A bias-0 upstream fixture family exists and gates the CPU path

- **Implementation state:** unimplemented
- **Document state:** draft
- **Principal failure reason:** the new fixtures encode a different model than the
  Rust side builds (the "pin every raw-dict default" trap), so the oracle is
  meaningless.
- **Scope:** `gen_bootstrap_dev()` + a `--bootstrap-dev-only` entrypoint in
  `crates/cb-oracle/generator/gen_fixtures.py`; new fixtures
  `crates/cb-oracle/fixtures/bootstrap_dev/{no,bayesian,bernoulli,mvs}/`; a new CPU
  oracle test `crates/cb-train/tests/bootstrap_dev_oracle_test.rs`.
- **Dependencies:** `catboost==1.2.10` `[VERIFIED: RUN]`; `ISOLATING_PARAMS`
  (`gen_fixtures.py:146-159`) which already pins `random_strength: 0`,
  `leaf_estimation_iterations: 1`, `score_function: "L2"`,
  `leaf_estimation_method: "Gradient"`, `random_seed: 0`, `thread_count: 1`; the
  existing frozen input `inputs/bootstrap_multiblock` (1500×4) — REUSED, not
  regenerated; the `--*-only` convention (`gen_fixtures.py:3246-3272`).
- **Input type:** the frozen `X.npy` / `y.npy`; per-scenario
  `{bootstrap_type, subsample | bagging_temperature}`.
- **Output type:** per scenario `model.json`, `staged.npy`, `predictions.npy`,
  `config.json`.
- **Preconditions:** the generator runs single-threaded; `boost_from_average=False`
  is the ONLY difference from the committed `bootstrap/` family.
- **Behavior:** every parameter is pinned identically on BOTH sides — Rust
  `BoostParams` must mirror `iterations=3, depth=2, learning_rate=0.1,
  l2_leaf_reg=3.0, random_strength=0.0, score_function=L2,
  leaf_method=Gradient, random_seed=0, boost_from_average=false`. The CPU oracle
  gates per-tree splits, per-tree leaf values, and per-iteration staged approximants
  at ≤1e-5, exactly as `bootstrap_oracle_test.rs` does for the bias-≠0 family.
- **Postconditions:** the committed `bootstrap/` fixtures are byte-unchanged
  (`git status` clean for that directory) and `bootstrap_oracle_test` stays green.
- **Errors:** the generator fails loudly if `catboost` is missing.
- **Acceptance examples:**
  - Given the new fixtures, when the CPU oracle runs, then all three sampled
    scenarios pass at ≤1e-5 with the CPU grower — proving the FIXTURE before any
    device claim rests on it.
  - Given `random_strength` is left unpinned on either side, then the oracle fails
    (the trap is actively detected, not just avoided).
  - Given `git status crates/cb-oracle/fixtures/bootstrap/`, then no file is
    modified.
- **Evidence:** `gen_fixtures.py:705-830` (`gen_bootstrap`), `:146-159`, `:3246-3272`;
  `bootstrap_oracle_test.rs:1-130`.
- **Non-goals:** a Poisson fixture (upstream rejects it on CPU,
  `gen_fixtures.py:718-722`).

---

### SPEC WR01-S15: The device reproduces upstream at ≤1e-5 for Bernoulli, Bayesian and MVS

- **Implementation state:** unimplemented
- **Document state:** draft
- **Principal failure reason:** an enabled bootstrap type does not meet the phase's
  parity bar against upstream CatBoost 1.2.10.
- **Scope:** `crates/cb-train/tests/device_bootstrap_fit_test.rs` (device-vs-CPU)
  and the device arm of `bootstrap_dev_oracle_test.rs` (device-vs-upstream).
- **Dependencies:** WR01-S1..S10 (the mechanism), WR01-S14 (the fixture),
  WR01-S11 (the base-path bar), WR01-S13 (the jitter budget).
- **Input type:** the `bootstrap_dev/{bernoulli,bayesian,mvs}` fixtures plus a
  synthetic ≥3-iteration fixture for the device-vs-CPU comparison.
- **Output type:** pass/fail + reported max deltas.
- **Preconditions:** rocm or cuda build; SKIP otherwise with an explicit message.
- **Behavior:** for each of the three types, assert (a) the fit is actually
  device-grown (an observable device signature, so a silent CPU fallback cannot
  pass), (b) `max|Δpred|` vs the upstream `predictions.npy` ≤ 1e-5, (c)
  `max|Δstaged|` vs `staged.npy` ≤ 1e-5, (d) `max|Δleaf|` vs the upstream
  `model.json` leaf values ≤ 1e-5, and (e) `max|Δpred|` vs the in-repo CPU grower
  ≤ 1e-5.
- **Postconditions:** no bootstrap type is device-eligible without a parity oracle
  behind it (D-04 / T-10-05).
- **Errors:** none.
- **Acceptance examples:**
  - Given Bernoulli `subsample = 0.8`, when the device fit is compared to upstream,
    then every gated quantity is within 1e-5.
  - Given the score/leaf channel split is reverted, then the leaf-value assertion
    fails first (the test is specific to the mechanism it protects).
  - Given a build where the gate silently declined, then the device-signature
    assertion fails rather than the test passing on CPU results.
- **Evidence:** `bootstrap_oracle_test.rs` (the comparison harness to mirror);
  `device_nonsym_fit_test.rs:136-149` (device-signature assertion precedent).
- **Non-goals:** Poisson (§8); weighted pools; `random_strength != 0`.

---

### SPEC WR01-S16: Poisson has one defined, backend-independent behaviour

- **Implementation state:** unimplemented
- **Document state:** draft
- **Principal failure reason:** a Poisson fit succeeds on one build and errors on
  another, or produces a device "model" with no upstream meaning.
- **Scope:** `device_host_eligible` (Poisson excluded by the allow-list, WR01-S9),
  the CPU dispatch error (`bootstrap.rs:435-438`), the session gate's Poisson arm,
  the stale Poisson comment at `session.rs:1567-1577`, and the device Poisson
  sampler self-oracle in `kernels/bootstrap_device_test.rs`.
- **Dependencies:** upstream supports Poisson on GPU only
  (`bootstrap_options.cpp:27-33`); this project's device grow reproduces CatBoost
  **CPU** semantics, NOT CatBoost GPU's algorithm; no CatBoost-CPU oracle can
  exist (`gen_fixtures.py:718-722`).
- **Input type:** `params.bootstrap_type == EBootstrapType::Poisson`.
- **Output type:** `CbResult<Model>` — always `Err`.
- **Preconditions:** none.
- **Behavior:** see §8 for the full decision and its rationale. In short:
  1. A Poisson `train()` request is rejected UP FRONT with a typed error whose
     message states (a) upstream supports Poisson only on GPU, (b) this project's
     device path targets CatBoost-CPU semantics, and (c) therefore no parity claim
     is possible — the SAME error under `cpu`, `wgpu`, `cuda` and `rocm` builds.
  2. The device Poisson sampler is RETAINED and its self-oracle strengthened:
     determinism (same `(seed, n, rate)` ⇒ bit-identical weights), finiteness,
     non-negativity, and agreement with an in-repo host Knuth reference — with an
     explicit module-doc statement that this is a capability oracle, not a parity
     oracle, and that Poisson is unreachable from `train()`.
  3. The session gate declines Poisson explicitly (defence in depth), so relaxing
     the host gate in a future phase cannot silently enable it.
- **Postconditions:** no build configuration makes a Poisson fit succeed;
  `bootstrap()`'s CPU error is never reached in a confusing "fell through to the CPU
  sampler" way, because the rejection happens before the boosting loop.
- **Errors:** `CbError::Degenerate` (DECIDED 2026-07-30 — `Unsupported` is the better
  semantic fit but changing the variant is breaking; keep `Degenerate`, fix the message —
  "unsupported on the active backend" is the semantically correct variant,
  `error.rs:86-92`) with the explanatory message above. **Note:** changing the
  variant is a behaviour change for any caller matching on `Degenerate`; see the
  task file for the compatibility check.
- **Acceptance examples:**
  - Given Poisson under the default `cpu` build, then `train` returns
    `Err(CbError::Degenerate(msg))` where `msg` names Poisson and GPU-only.
  - Given Poisson under a `rocm` build, then the SAME error variant and message.
  - Given the device Poisson sampler run twice with one seed, then bit-identical
    weights, all finite, all `>= 0.0`.
- **Evidence:** `bootstrap.rs:40-43, 435-438`; `session.rs:1560-1579`;
  `error.rs:86-92`; `gen_fixtures.py:718-722`.
- **Non-goals:** any ≤1e-5-vs-upstream claim for Poisson; implementing
  CatBoost-GPU tree-growing semantics.

---

## 6. Acceptance scenarios (phase level)

| # | Given | When | Then |
|---|---|---|---|
| AC-1 | any fit with `bootstrap_type = No` | trained on a device build | splits, `leaf_of`, `part_stats` are bit-identical to `5a5068a` (D-04) |
| AC-2 | the three shapes of WR01-S11 | device vs CPU | `max\|Δpred\| ≤ 1e-5`; split mismatches reported, attributed |
| AC-3 | `bootstrap_dev/{bernoulli,bayesian,mvs}` | CPU grower vs upstream | ≤1e-5 on splits, leaves, staged, predictions |
| AC-4 | the same fixtures | device grower vs upstream | ≤1e-5, with a device-signature assertion |
| AC-5 | the same fixtures | device grower vs CPU grower | ≤1e-5 |
| AC-6 | a Bayesian device fit ×5 | repeated in one process | identical splits, `max\|Δpred\| ≤ 1e-7` |
| AC-7 | `bootstrap/` (bias≠0) family | `cargo test -p cb-train --test bootstrap_oracle_test` | green, unchanged (blocking) |
| AC-8 | Poisson | any build | one typed `Degenerate` error, identical everywhere |
| AC-9 | a synthetic over-range `s` | grow requested | typed `OutOfRange`, no launch |

---

## 7. Verification strategy and where each check runs

| Check | Command | Where |
|---|---|---|
| CPU no-regression (blocking) | `cargo test -p cb-train --test bootstrap_oracle_test` | anywhere |
| CPU no-regression (draw model) | `cargo test -p cb-train --test regularization_oracle_test` | anywhere |
| CPU full suite | `cargo test -p cb-train` (known pre-existing red: `monotone_non_symmetric_and_region_are_typed_errors`) | anywhere |
| draw-replay unit | `cargo test -p cb-train --lib device_draw_replay` | anywhere |
| new bias-0 CPU oracle | `cargo test -p cb-train --test bootstrap_dev_oracle_test` | anywhere |
| session host-sample unit | `cargo test -p cb-backend --no-default-features --features rocm --lib host_sample -- --test-threads 1` | **local ROCm** |
| device sampler self-oracles | `cargo test -p cb-backend --no-default-features --features rocm --lib bootstrap` / `--lib mvs` | **local ROCm** |
| base-grower ≤1e-5 oracle | `cargo test -p cb-train --no-default-features --features rocm --test device_oblivious_parity_test -- --nocapture` | **local ROCm** |
| tie-break characterisation | `... --test device_split_tiebreak_test -- --nocapture` | **local ROCm** |
| device bootstrap parity | `... --test device_bootstrap_fit_test -- --nocapture` | **local ROCm** |
| device upstream parity | `... --test bootstrap_dev_oracle_test -- --nocapture` | **local ROCm** |
| seam no-regression | `... --test device_nonsym_fit_test`, `... --test device_region_fit_test`, `... --test device_seam_test` | **local ROCm** |
| CUDA sign-off | the same four device targets with `--features cuda` | **Kaggle P100** |

**Build landmines (both VERIFIED in research):**
1. Always `--no-default-features --features rocm` — `cb-train`'s `default = ["cpu"]`
   otherwise unifies cpu in `[VERIFIED: crates/cb-train/Cargo.toml:15-20]`.
2. A blanket `cargo test -p cb-train --no-default-features --features rocm` does NOT
   build (37 test files `use cb_backend::CpuBackend`, which is not compiled under
   rocm) — always select `--test <target>` `[RESEARCH, VERIFIED there]`.

**Tolerance discipline.** ε = **1e-5** for every sign-off assertion in this phase.
The shipped ε=1e-4 device precedents (`device_nonsym_fit_test.rs:167`,
`bootstrap_device_test.rs:25`, `mvs_device_test.rs:26`) must NOT be copied. WR01-S13
uses the tighter 1e-7 deliberately (a jitter budget, not a parity bar).

---

## 8. Poisson: the resolved behavioural decision

**Question posed:** should a Poisson fit now SUCCEED on device while the CPU path
errors, and what happens when Poisson is requested but the device declines?

**Decision (D5): No. Poisson is rejected up front, identically on every backend.**

Rationale:

1. **A device Poisson model would be a fabricated result.** Upstream supports
   Poisson only on GPU, and CatBoost-GPU's tree-growing algorithm is *not* what this
   project's device path reproduces — our device grow mirrors CatBoost **CPU**
   semantics (that is the whole basis of every device oracle in the repo). A Poisson
   device fit would therefore produce a model that matches neither CatBoost CPU (it
   errors) nor CatBoost GPU (different algorithm). That is exactly the
   "never a fabricated device result" rule (D-04 / T-10-05, restated at
   `boosting.rs:3044-3053`).
2. **Feature-dependent success is the worst possible UX.** If Poisson succeeded
   under `--features rocm` and errored under `--features cpu`, the same Python or
   Rust program would train or fail depending on which wheel was installed, with no
   diagnostic. The user's own constraint — "must not silently fall back into the CPU
   error path in a confusing way" — is best served by ONE deterministic answer.
3. **The "confusing fallback" is fixed by moving the rejection earlier.** Today a
   Poisson request walks into the boosting loop and only fails inside `bootstrap()`
   on tree 0 with `Degenerate("poisson bootstrap is not supported on CPU")` — a
   message that is misleading on a GPU build. The fix is to reject at parameter
   validation, before any device session or gradient work, with
   `CbError::Degenerate` and a message that explains *why* (GPU-only upstream;
   this project targets CPU semantics; no parity oracle possible).
4. **The capability is preserved and measured, not deleted.** The device Poisson
   sampler stays, with a strengthened self-oracle covering exactly the criteria the
   user asked for — determinism (same seed ⇒ identical output), agreement with an
   in-repo host Knuth reference, and finiteness/sanity — plus explicit documentation
   that it intentionally diverges from the CPU path (which errors) and carries no
   parity claim. That keeps a future "CatBoost-GPU-semantics" phase cheap.

**If the user overrides this decision** and wants a runnable device Poisson fit,
`plan9.md` §"Alternative A" specifies the minimal delta: add `Poisson` to the
WR01-S9 allow-list *only* under an explicit opt-in parameter (never by default),
keep the sampler on the device (Design B for this arm only, since there is no CPU
sampler to host-run), fix the arbitrary stream advance at `session.rs:1567-1577`
to be consumption-faithful, and label every resulting model as
non-upstream-comparable in the model metadata. That path is scoped but NOT
scheduled in this phase.

---

## 9. Risks and guardrails

| # | Risk | Prevention | Verification |
|---|---|---|---|
| R1 | Sampled leaves (the latent bug) | WR01-S2 lands before the gate opens (task order) | leaf assertion in WR01-S15; `Σ weight == n` unit test |
| R2 | Unsampled der in the score histogram | WR01-S1 | hand-built `s` reference test |
| R3 | RNG phase drift | WR01-S7 with a literal `std_normal` call | `raw_state()` + `call_count()` equality unit test |
| R4 | MVS λ frozen | WR01-S8 | ≥3-iteration MVS scenario |
| R5 | Fixtures not device-eligible | WR01-S14 (`boost_from_average=False`) | device-signature assertion in WR01-S15 |
| R6 | Silent CPU fallback passes a "device" test | device-signature assertion + `CpuRefRuntime` diff | WR01-S11 / S15 |
| R7 | cpu/rocm feature unification | always `--no-default-features` | §7 |
| R8 | Blanket rocm test build fails | always `--test <target>` | §7 |
| R9 | ε=1e-4 creep | ε pinned at 1e-5 in every new assertion | code review + §7 |
| R10 | **Split tie flipped on a non-degenerate split** (F-B) | WR01-S12 characterisation with a gain-gap floor assertion | tie-break test |
| R11 | **Fixed-point range overflow under MVS** (F-D) | WR01-S10 host guard | over-range unit test |
| R12 | **Leaf-reduce float-atomic jitter** (F-E) | WR01-S13 budget with a documented escalation | ×5 determinism test |
| R13 | Fixture regeneration clobbers committed fixtures | `--bootstrap-dev-only` targeted entrypoint; new directory | `git status` assertion in the task's Verify step |
| R14 | Per-tree `8·n` upload cost regresses GPU speed | measure with `CB_GPU_PROF=1`; Design B′ deferred | non-gating bench note in `plan8.md` |
| R15 | Poisson variant change breaks a `Degenerate` matcher | grep all match sites before changing the variant | WR01-S16 task step |

---

## 10. Traceability

| Spec | Requirement source | Code evidence | Task |
|---|---|---|---|
| WR01-S1 | phase goal; research A4 | `mod.rs:3836-3852` | TASK-03 |
| WR01-S2 | research finding 10/11 | `mod.rs:3942-3970` | TASK-03 |
| WR01-S3 | D-04 | `mod.rs:3757-4009` | TASK-03 |
| WR01-S4 | research A2 | `runtime.rs:1279-1286` | TASK-04 |
| WR01-S5 | D4 (explicit flag) | `session.rs:1038-1092` | TASK-04 |
| WR01-S6 | research A1 step 4 | `boosting.rs:3738-3770` | TASK-06 |
| WR01-S7 | research A1 steps 1/6/7 | `tree.rs:610-614`, `:1142-1218` | TASK-05 |
| WR01-S8 | research A1 step 8 | `boosting.rs:4586-4589` | TASK-05 |
| WR01-S9 | phase goal | `boosting.rs:3073`, `:3133-3160` | TASK-06 |
| WR01-S10 | F-D (new) | `kernels.rs:2325-2335` | TASK-03 |
| WR01-S11 | F-A (new) | probe file | TASK-01 |
| WR01-S12 | F-B/F-C (new) | `mod.rs:3069-3078`, `tree.rs:1175-1204` | TASK-02 |
| WR01-S13 | F-E (new) | `kernels.rs:3800-3826` | TASK-10 |
| WR01-S14 | user decision 2 | `gen_fixtures.py:705-830` | TASK-07 |
| WR01-S15 | phase goal | `bootstrap_oracle_test.rs` | TASK-08 |
| WR01-S16 | user decision 3 + D5 | `bootstrap.rs:435-438` | TASK-09 |

---

## 11. Open items requiring a user decision

1. **D5 (Poisson) — RESOLVED 2026-07-30: ACCEPTED.** Reject up front, uniformly on
   every backend. `plan9.md`'s "Alternative A" is dead. Empirically backed by the
   Kaggle P100 run: official CatBoost CPU rejects Poisson, official CatBoost GPU
   trains it — so it is GPU-only upstream and has no CPU-semantics parity target.
2. **`CbError::Degenerate → Unsupported` — RESOLVED 2026-07-30: REJECTED.** Keep
   `CbError::Degenerate`; fix only the message. No caller-visible variant change.
3. **WR01-S13 escalation.** If the ×5 determinism budget (≤1e-7) fails, converting
   `partition_update_kernel` to fixed-point `Atomic<u64>` is a bounded but
   non-trivial addition to the phase. It is planned as a conditional follow-on
   (`plan10.md` §Escalation), not scheduled up front.

---

## 12. TreeFinder synchronization status

TreeFinder MCP is available in this session, but this repository has **no**
TreeFinder-indexed specification store: all twelve prior phases keep their spec of
record as a plain `.planning/plans/<slug>/SPEC.md` file, and no document exists in
the TreeFinder corpus for any of them. Registering only this phase would fork the
specification store and break the sibling convention.

**Action taken:** this file is the draft spec of record (`document_state: draft`,
all specs `implementation_state: unimplemented`). No TreeFinder document was
created or updated, so nothing was added, updated, or left stale there.

**To adopt TreeFinder for this corpus** (a separate, opt-in migration): register all
thirteen `SPEC.md` files under the `catboost-rs/planning/` prefix in one pass, then
switch this header's `treefinder_pending` block to the real document identifier.
