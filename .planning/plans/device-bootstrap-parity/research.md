# Phase Research: Device Bootstrap Parity (WR-01 wiring)

Branch: `fix/bootstrap-rng-draw-accounting` @ `5a5068a` (working tree clean except untracked
`bench/bootstrap_gpu/`) [VERIFIED: `git status --short`, `git log --oneline -3`].
Research only — no production code was modified.

---

## Research Summary

- **Phase goal.** Make `bootstrap_type != No` fits eligible for the GPU/device tree grower while
  holding the project's ≤1e-5 parity bar against upstream CatBoost 1.2.10. This is the work the
  codebase defers as "WR-01" at `crates/cb-train/src/boosting.rs:3143-3159`
  [PROJECT: crates/cb-train/src/boosting.rs:3143].
- **Recommended approach: Design A (host-computed sample, per-tree multiplier upload), with a
  mandatory device-side score/leaf channel split.** Keep `bootstrap()` and the whole RNG draw
  model on the host exactly as committed in `5a5068a`; upload one `n`-length multiplier
  `s[i] = control[i] ? sample_weights[i] : 0.0` per tree; make the device grow use
  `(der1·s, weight·s)` for the split histogram and the UNSAMPLED `(der1, weight)` for leaf
  estimation. Design B (device-resident sampling) is a viable *later perf* refinement — its
  samplers are already measurably accurate — but it does not remove any of Design A's required
  device work and it adds the two-stream RNG hazard back.
- **Most important constraints.**
  1. Under sampling the CPU splits the score path from the leaf path
     (`boosting.rs:3723-3770`); the device grow does **not** — it feeds the SAME
     `(der1, weight_ref)` pair into both the histogram fill and `launch_partition_update_into`
     (`crates/cb-backend/src/gpu_runtime/mod.rs:3836-3970`). Wiring the existing device bootstrap
     arm as-is would produce a numerically WRONG model, independent of RNG phase.
  2. The device path `continue`s at `boosting.rs:3486`, consuming **zero** RNG draws and never
     updating `prev_leaf_mean_l2`. Since `5a5068a`, an active-sampling host tree consumes
     `2 + bootstrap() + Σ_levels(n_features GenRandReal1 + 1 GenRand + n_features std_normal) + 2`
     draws, and the `std_normal` count is **data-dependent** (Marsaglia-polar rejection)
     [PROJECT: .planning/plans/bayesian-rng-draw-accounting/instrumented-ground-truth/GROUND_TRUTH.md:60-71].
     The device branch must replay these literally.
  3. The committed bootstrap oracle fixtures all use `boost_from_average=True`, so `bias != 0`
     and the CR-01 gate (`boosting.rs:3092`) excludes them from the device path **even after
     the bootstrap gate is relaxed**. The existing fixtures alone cannot gate device parity.
- **Highest-risk findings.**
  - The score-vs-leaf channel divergence (above) — the single largest piece of work, and it is a
    latent correctness bug in the *already-landed but unwired* device bootstrap arm.
  - MVS λ is pinned once per fit in `DeviceTrainConfig.mvs_lambda`
    (`crates/cb-compute/src/runtime.rs:1096`, consumed at `session.rs:1592`), but upstream λ
    changes every tree (`CalculateLastIterMeanLeafValue`). Structural gap for any device MVS.
  - Poisson is unreachable at ≤1e-5: upstream rejects it on CPU and the in-repo CPU `bootstrap()`
    returns `CbError::Degenerate` (`crates/cb-train/src/bootstrap.rs:435-438`). Enabling it on the
    device would make the device path *succeed where the CPU path errors* — a behavioural
    divergence, not a parity win.

---

## Phase Requirements

### In Scope

- Relax `device_host_eligible` (`crates/cb-train/src/boosting.rs:3054-3098`) so a
  `bootstrap_type ∈ {Bayesian, Bernoulli, Mvs}` fit can reach the device grower
  [CODEGRAPH: `device_host_eligible`; PROJECT: boosting.rs:3073].
- Thread the sampling knobs from `params` instead of leaving them at `DeviceTrainConfig::default()`
  (`boosting.rs:3133-3160`).
- Make the device grow numerically reproduce the CPU sampling semantics
  (`boosting.rs:3723-3770`): sample affects the split histogram only; leaves use unsampled der/weight.
- Keep the host RNG stream phase-exact on the device path (replay the per-level / post-tree draws).
- Add oracle tests proving device == CPU (and, where fixtures permit, device == upstream) at ≤1e-5.

### Acceptance Criteria

- `cargo test -p cb-train --test bootstrap_oracle_test` stays green (CPU no-regression, D-04)
  — this is the blocking gate the bench script already treats as blocking
  [PROJECT: bench/bootstrap_gpu/bootstrap_bench.py:71-80].
- A new device-vs-CPU end-to-end parity test at **≤1e-5** for each newly-eligible bootstrap type,
  running under `--no-default-features --features rocm` on the local gfx1151 rig and SKIPping
  cleanly elsewhere.
- No `bootstrap_type` becomes device-eligible without a parity oracle behind it (the D-04 /
  T-10-05 "never a fabricated device result" rule, restated at `boosting.rs:3044-3053`).
- Poisson continues to route to the CPU path (and therefore to the existing typed error).

### Out of Scope

- `random_strength != 0` on the device. The device grower has no perturbation arm; the
  `SelectBestCandidate` normal instance is only a numeric no-op at `score_st_dev == 0`
  [PROJECT: boosting.rs:3815-3833]. Keep `params.random_strength == 0.0` in the gate
  (`boosting.rs:3074`).
- Weighted pools (`weights.iter().all(|&w| w == 1.0)`, `boosting.rs:3086`, WR-03 upstream
  `Σ w·der` gap) — orthogonal and independently blocking.
- Lifting CR-01 (`bias == 0.0`, `boosting.rs:3092`) *unless* the phase chooses the
  "reuse existing fixtures" testing route (see Testing).
- Non-symmetric / Region / CTR / exact-leaf × bootstrap combinations. Every device gate is
  all-or-nothing PER family (`session.rs:1058-1074`, `:1093-1105`).

### Open or Conflicting Requirements

- The user's phase framing says "the eligibility gate relaxed"; the code's own WR-01 note says the
  same, but does NOT state which bootstrap types are targeted. This research recommends
  `{Bernoulli, Bayesian, Mvs}` in, `Poisson` out (evidence below). Needs an explicit decision.
- Doc/code conflict (harmless but must not be trusted): `bootstrap_device.rs:156-157` and
  `bootstrap_device_test.rs:34-36` say the device Bayesian kernel "uses the exact `ln`". The
  kernel actually calls a `#[cube]` transcription of upstream `FastLog2f`
  (`crates/cb-backend/src/kernels/bootstrap_device.rs:104-111`, called at `:230`)
  [PROJECT: crates/cb-backend/src/kernels/bootstrap_device.rs:104,230]. Prefer the code.

---

## Project Constraints

- **Parity bar ≤1e-5** against upstream CatBoost outputs [PROJECT: CLAUDE.md "Constraints"].
  Device-side self-oracles in this repo use a looser ε=1e-4 "GPU bar"
  (`bootstrap_device_test.rs:25-26`, `mvs_device_test.rs:26-27`,
  `device_nonsym_fit_test.rs:167`). **A WR-01 sign-off test must not silently inherit ε=1e-4** —
  the phase goal is explicitly ≤1e-5.
- **Source/test separation is mandatory**: no `#[cfg(test)] mod tests` inline. In `cb-backend`
  the convention is a sibling `*_test.rs` mounted from the parent module under `#[cfg(test)]`
  (`crates/cb-backend/src/kernels.rs:2941-2946` mounts `kernels/bootstrap_device_test.rs`).
  In `cb-train` the convention is an integration test under `tests/`
  [PROJECT: CLAUDE.md "Source/Test Separation"; crates/cb-train/tests/*].
- **Workspace lints deny** `unwrap_used`, `expect_used`, `panic`, `indexing_slicing`
  [VERIFIED: Cargo.toml:10-14]. Test files opt out with a file-level
  `#![allow(clippy::unwrap_used, …)]` (e.g. `device_nonsym_fit_test.rs:19`).
- **No `cubecl` type may appear in `cb-compute`** (`DeviceTrainConfig` doc,
  `crates/cb-compute/src/runtime.rs:1071-1081`); **no `cb-train` dependency from `cb-backend`**
  (RNG semantics are transcribed inline, `session.rs:108-110`,
  `crates/cb-backend/src/kernels.rs:2935-2939`).
- **Backend selection is compile-time only** (`crates/cb-backend/Cargo.toml [features]`), and
  `cb-train` forwards it with `default-features = false` to avoid cpu/rocm unification
  [VERIFIED: crates/cb-train/Cargo.toml features block].
- **CubeCL rules** (from CLAUDE.md): kernels must be generic-float; read the CubeCL manual at
  `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` before writing kernel code;
  on a CubeCL build error load `cubecl_error_guideline.md` first. (Design A needs **no new
  kernel**, see below — this constraint is mostly inert for the recommended path.)
- **Locked decision from `5a5068a`**: the host per-tree draw model is now instrumented-ground-truth
  backed. It must be treated as authoritative and reused, never re-derived
  [PROJECT: .planning/plans/bayesian-rng-draw-accounting/instrumented-ground-truth/GROUND_TRUTH.md].

---

## Current Project Architecture

### Relevant subsystems and boundaries

| Layer | Path | Role here |
|---|---|---|
| `cb-train` boosting loop | `crates/cb-train/src/boosting.rs` | owns the persistent RNG, `bootstrap()` call, eligibility gate, device branch |
| `cb-train` sampler | `crates/cb-train/src/bootstrap.rs` | CPU `bootstrap()` / Bayesian / Bernoulli / MVS, upstream-transcribed |
| `cb-train` grower | `crates/cb-train/src/tree.rs` | `greedy_tensor_search_oblivious_perturbed`, `Perturbation`, per-level draws |
| `cb-compute` seam | `crates/cb-compute/src/runtime.rs` | `Runtime` trait, `DeviceTrainConfig`, `DeviceBootstrapType` (plain host types) |
| `cb-backend` facade | `crates/cb-backend/src/gpu_backend.rs` | `GpuBackend`, holds the `RefCell<Option<GpuTrainSession>>` |
| `cb-backend` session | `crates/cb-backend/src/gpu_runtime/session.rs` | `GpuTrainSession::begin` gate, `grow_one`, `BootstrapState`, `MvsState` |
| `cb-backend` grow | `crates/cb-backend/src/gpu_runtime/mod.rs` | `grow_oblivious_tree_resident` (histogram / split / part-stats / leaf) |
| `cb-backend` kernels | `crates/cb-backend/src/kernels/{bootstrap_device,mvs_device}.rs` | device samplers + `fold_weights_resident` |
| `cb-oracle` | `crates/cb-oracle/{fixtures,generator}` | frozen upstream fixtures + `gen_fixtures.py` |

### Existing data / control flow (device path, today)

1. `train_inner` computes `weights`, `sum_all_weights = sum_f64(&weights)` **once**
   (`boosting.rs:2371-2376`).
2. `device_host_eligible` is decided once per fit (`boosting.rs:3054-3098`).
3. `device_scaled_l2 = scale_l2_reg(params.l2_leaf_reg, sum_all_weights, n)` once
   (`boosting.rs:3104`) — matches the CPU per-tree `scaled_l2` at `boosting.rs:3802`, which reads
   the same tree-invariant `sum_all_weights`. **Sampling does not change this**: the CPU per-tree
   L2 scaling ignores `score_weights` entirely (`boosting.rs:3787-3802`). *Finding 8 confirmed,
   HIGH confidence.*
4. `device_config` is built from `params.grow_policy` / `max_leaves` / `min_data_in_leaf` only;
   `bootstrap_type`, `mvs_lambda`, `exact_leaf`, `ctr`, `sample_rate`, `rng_seed` are left at
   `DeviceTrainConfig::default()` (`boosting.rs:3133-3160`). *Finding 2 confirmed.*
5. `runtime.begin_device_training(...)` uploads the cindex + `weight` + scalars ONCE
   (`boosting.rs:3162-3176`; trait at `runtime.rs:1230-1262`; impl at `gpu_backend.rs:250`).
6. Per iteration, `runtime.grow_tree_on_device(&approx, target)` — **no per-tree weight/sample
   argument** (`runtime.rs:1279-1286`, `gpu_backend.rs:310-332`). *Finding 9 confirmed.*
7. `GpuTrainSession::grow_one` chains der1 on-device, optionally draws a bootstrap/MVS sample and
   folds it into a per-tree weight, then calls `grow_oblivious_tree_resident`
   (`session.rs:1546-1674`).
8. The `cb-train` device branch folds the tree, updates `approx` on the host, records `staged_out`,
   and `continue`s (`boosting.rs:3319-3400`, `:3486`). **No RNG draw, no `prev_leaf_mean_l2`.**

### Existing reusable implementations (do not reimplement)

- `cb_train::bootstrap::bootstrap` (`bootstrap.rs:383-439`) — the ≤1e-5-verified CPU sampler,
  including the FastLogf-exact Bayesian weight (`bootstrap.rs:106-152`), the f32-rounded Bernoulli
  threshold (`bootstrap.rs:176-188`), and the MVS quickselect threshold (`bootstrap.rs:206-300`)
  plus MVS's two `performRandomChoice=false` compensation draws (`bootstrap.rs:413-423`).
- `cb_train::bootstrap::last_iter_mean_leaf_value` (`bootstrap.rs:363-369`) — per-tree MVS λ input.
- `cb_core::TFastRng64` (`crates/cb-core/src/rng.rs`), including `raw_state()` (`:221`) and
  `advance()` (`:192`), and `cb_core::std_normal` (`crates/cb-core/src/normal.rs:50`,
  re-exported `crates/cb-core/src/lib.rs:24`).
- `cb_backend::kernels::bootstrap_device::fold_weights_resident`
  (`bootstrap_device.rs:417-439`) — a **generic elementwise f64 product** built on
  `crate::kernels::vector_mul_kernel`. Reusable verbatim for `der1 · s` as well as `weight · s`.
  **No new kernel is required for Design A.**
- `launch_bootstrap_weights_resident` / `launch_mvs_weights_resident` — the device samplers
  (needed only for Design B).
- `cb_compute::calc_average` (`crates/cb-compute/src/leaf.rs:83-89`) and
  `cb_compute::scale_l2_reg`.
- `accumulate_leaf_weights` / `normalize_leaf_values` — already used by the device fold arm
  (`boosting.rs:3374-3383`).

### Current conventions and patterns

- **Coverage-gate pattern A**: an `Option`-returning `map_*` classifier plus an all-or-nothing
  per-family boolean in `GpuTrainSession::begin` (`session.rs:1043-1091`).
- **Device parity test pattern (cb-train integration)**: a `#[cfg(any(feature="rocm", feature="cuda"))]
  mod device { … }` containing a local `CpuRefRuntime` that inherits the trait defaults
  (`begin → Ok(false)`, `grow → Ok(None)`) so `train()` runs the CPU grower, plus a
  `#[cfg(not(...))] { eprintln!("SKIP …") }` else-arm
  [PROJECT: crates/cb-train/tests/device_nonsym_fit_test.rs:92-122, 182-202]. `CpuBackend` is
  **not compiled** under rocm/cuda (`crates/cb-backend/src/lib.rs:33-37`), which is why the
  reference runtime is hand-rolled in the test.
- **Device capability skip pattern (cb-backend unit)**: `fn device_backend_active() -> bool {
  cfg!(any(feature = "rocm", feature = "cuda")) }` (e.g. `bootstrap_device_test.rs:29-31`), plus a
  runtime `Err(CbError::Unsupported(msg)) if msg.contains("Atomic<u64>")` early-return
  (`session_depth_gt1_test.rs:660-670`).

---

## Standard Stack

| Item | Version / state | Existing or proposed | Purpose this phase | Constraints | Project usage | Authoritative finding |
|---|---|---|---|---|---|---|
| Rust workspace, edition 2021 | latest stable | existing | all | workspace lints deny unwrap/panic/indexing | all crates | [VERIFIED: Cargo.toml:10-14] |
| `cubecl` | `0.10.0` | existing | device kernels | backend selected via `cb-backend` features only; `cubecl/hip` for rocm | `cb-backend` only | [VERIFIED: Cargo.toml:38; crates/cb-backend/Cargo.toml features] |
| ROCm / HIP (`gfx1151`) | AMD Radeon 860M, `amdgcn-amd-amdhsa--gfx1151` | existing local rig | run device parity tests locally | `cubecl-hip-sys` autodetects HIP; never pin `hip_xxx` features | `--no-default-features --features rocm` | [VERIFIED: `rocminfo`; successful `cargo test -p cb-backend --no-default-features --features rocm`] |
| `cb-core` RNG (`TFastRng64`, `std_normal`) | in-repo | existing | host draw replay on the device branch | bit-for-bit upstream transcription | `boosting.rs`, `tree.rs`, `cb-compute` | [CODEGRAPH: `TFastRng64`, `std_normal`] |
| `bytemuck` | `1` | existing | host↔device casts | `cb-backend` only | `session.rs`, kernels | [VERIFIED: Cargo.toml:39] |
| `rayon` | workspace | existing | host per-feature parallelism | order-preserving only | `tree.rs`, `cb-backend` prep | [VERIFIED: Cargo.toml] |
| `cb-oracle` fixtures (upstream CatBoost 1.2.10) | frozen | existing | ≤1e-5 upstream gate | `bootstrap/{no,bayesian,bernoulli,mvs}` only; all `boost_from_average=True` | `bootstrap_oracle_test.rs` | [VERIFIED: `ls crates/cb-oracle/fixtures/bootstrap/*/`; gen_fixtures.py:784-797] |

**Context7 CLI note.** No external library documentation is load-bearing for this phase: every
dependency in the change surface (`cubecl`, `bytemuck`, `rayon`, `ndarray-npy`) is already wired
and no new API is proposed. Context7 was therefore not invoked; if the planner adds a dependency,
resolve it with `npx --yes ctx7@latest library "<name>"` then `npx --yes ctx7@latest docs "<id>"`
before pinning. `[ASSUMED: no new external API surface]` — see Open Questions.

---

## Dependency Analysis

- **Direct crate dependencies changed:** none. The phase is internal plumbing across
  `cb-train → cb-compute → cb-backend`.
- **Internal dependency direction (must be preserved):** `cb-backend` depends on `cb-compute` and
  `cb-core`; it must **never** depend on `cb-train` (RNG/sampler semantics are transcribed, see
  `session.rs:108-110`). Design A keeps the sampler in `cb-train`, so this constraint is trivially
  satisfied. Design B keeps it satisfied too, but only because the samplers were already
  transcribed.
- **`DeviceTrainConfig` is a plain-host struct** in `cb-compute` with 20 call sites
  [CODEGRAPH: `DeviceTrainConfig` blast radius: `crates/cb-compute/src/lib.rs`,
  `crates/cb-train/src/boosting.rs`, `crates/cb-compute/src/runtime.rs`,
  `crates/cb-backend/src/gpu_backend.rs`, `crates/cb-backend/src/gpu_runtime/session.rs`]. Any new
  field is additive and safe (`..DeviceTrainConfig::default()` is used at call sites).
- **`Runtime::grow_tree_on_device` has exactly 3 implementations**: the trait default
  (`runtime.rs:1279`), `GpuBackend` (`gpu_backend.rs:310`), and a test `FakeRuntime`
  (`crates/cb-train/tests/device_seam_test.rs:69`) [VERIFIED: `grep -rn "fn grow_tree_on_device"`].
  A signature change is a 3-site + 1-caller change (`boosting.rs:3198`).
- **Runtime / build / system dependencies:** ROCm (HIP) runtime for the parity tests; `Atomic<u64>`
  device capability for the resident partition histogram (the session emits
  `CbError::Unsupported("… Atomic<u64> …")` when absent, `session_depth_gt1_test.rs:660-670`).
  The local gfx1151 rig **does** run the resident bootstrap grow: `session_bootstrap_grows_finite_tree`
  passed, not skipped [VERIFIED: `cargo test -p cb-backend --no-default-features --features rocm --lib bootstrap`].
- **wgpu is structurally excluded** from every bootstrap path: `wgpu_reject()`
  (`bootstrap_device.rs:288-294`) — no f64/u64 channels.
- **Additions / removals:** none proposed.

---

## Recommended Architecture and Implementation Pattern

### Prescribed approach — Design A, in four layers

**A1. `cb-train` gate + config threading (`crates/cb-train/src/boosting.rs`).**

- `device_host_eligible` (`:3054-3098`): replace `&& matches!(params.bootstrap_type, EBootstrapType::No)`
  (`:3073`) with `&& !matches!(params.bootstrap_type, EBootstrapType::Poisson)`. Keep `:3074`
  (`random_strength == 0.0`), `:3086` (unit weights), `:3092` (`bias == 0.0`), `:3098`
  (Gradient/Simple leaf) unchanged.
- `device_config` (`:3133-3160`): thread `bootstrap_type` (mapped `EBootstrapType → DeviceBootstrapType`),
  `sample_rate = params.subsample as f32`, `rng_seed = params.random_seed`. **Do NOT** thread
  `mvs_lambda` under Design A — the device sampler is unused; leaving these at default keeps
  `is_covered_regime()` true so the session takes the byte-unchanged non-sampling arm.
  *(Alternative: add an explicit `DeviceTrainConfig::sample_from_host: bool` so the session's
  intent is unambiguous rather than implied by "config says No but a sample arrives". Recommended —
  it prevents a future reader from wiring both samplers at once.)*
- Device branch (`:3196-3486`), when `draws_active`:
  1. `for _ in 0..PRE_TREE_DRAWS { rng.gen_rand(); }` — mirroring `:3667-3671`.
  2. Build `der_obj` **only for MVS** (it is the only arm reading `derivatives`,
     `bootstrap.rs:410-412`). The host owns `approx` on this branch, so
     `runtime.compute_gradients(&params.loss, &approx, target, 1)` reproduces the CPU
     `weighted_der1` (unit weights ⇒ `weighted_der1 == der1`, cf. `boosting.rs:3640-3645`).
     Bernoulli/Bayesian pass a length-`n` placeholder, avoiding the extra round trip.
  3. `let sampled = bootstrap(params.bootstrap_type, &der_obj, params.subsample,
     params.bagging_temperature, prev_leaf_mean_l2, &mut rng)?;`
  4. `let s: Vec<f64> = (0..n).map(|i| if control[i] { sample_weights[i] } else { 0.0 }).collect();`
     — exactly the CPU `score_weights` at unit weight (`boosting.rs:3758-3770`), and exactly the
     multiplier for `score_weighted_der1` (`:3738-3751`).
  5. `runtime.grow_tree_on_device(&approx, target, &s)`.
  6. After the fold, replay the grow draws that the device did not consume:
     `for _ in 0..params.depth { for _ in 0..matrix.n_features() { rng.gen_rand_real1(); }
      rng.gen_rand(); for _ in 0..matrix.n_features() { let _ = std_normal(&mut rng); } }`
     — mirroring `tree.rs:610-614`, `tree.rs:1142`, and the per-feature `GetInstance` pass
     (`tree.rs:1149+`). **Must call `std_normal` literally**, not a formula: its `gen_rand` count is
     data-dependent [PROJECT: GROUND_TRUTH.md:60-71].
  7. `for _ in 0..POST_TREE_EXTRA_DRAWS { rng.gen_rand(); }` — mirroring `:4580-4584`.
  8. `prev_leaf_mean_l2 = Some(last_iter_mean_leaf_value(&device_leaf_values));` — mirroring
     `:4586-4589`. Required for MVS trees ≥ 1; harmless otherwise.
  - **Ordering note.** On the CPU path the RSM / CalcScores / SelectBestCandidate draws happen
    *during* the grow, i.e. between `bootstrap()` and the leaf phase. Because the device grow draws
    nothing, replaying them in one block after the grow yields the identical stream position for the
    next tree. Verify against the CPU stream with a direct `call_count()` comparison test.

**A2. Seam widening (`crates/cb-compute/src/runtime.rs`).**

```
fn grow_tree_on_device(&self, approx: &[f64], target: &[f64], sample: &[f64])
    -> CbResult<Option<DeviceGrownTree>>
```
Empty `sample` ⇒ the byte-unchanged no-sampling path (D-04). Default impl binds and returns
`Ok(None)`.

**A3. Session (`crates/cb-backend/src/gpu_runtime/session.rs::grow_one`, `:1449+`).**

- Upload `sample` (`upload_channel_floats`, the existing helper) into a per-tree handle.
- `score_weight_h = fold_weights_resident(&client, &self.weight_h, &sample_h, n)?`
- `score_der1_h  = fold_weights_resident(&client, &der1_h,       &sample_h, n)?`
  — the same generic `vector_mul_kernel` (`bootstrap_device.rs:417-439`); no new kernel.
- Pass **four** handles to the grow: `(score_der1_h, score_weight_h)` for the histogram and
  `(der1_h, self.weight_h)` for leaf estimation.
- Leave `BootstrapState` / `MvsState` unreached from `train()`. Their `#[cfg(test)]` self-oracles
  stay green; add a comment replacing the WR-01 note so the deferral status is accurate.

**A4. Grow (`crates/cb-backend/src/gpu_runtime/mod.rs::grow_oblivious_tree_resident`, `:3757`).**

- Add `score_der1_h` / `score_weight_h` parameters.
- `launch_partition_hist2_resident_into` (`:3836-3852`) consumes the **score** pair.
- `launch_partition_update_into` (`:3942-3951`) keeps the **unsampled** `der1_h` / `weight_h` so
  `calc_average(Σ der1, Σ weight, scaled_l2)` at `:3963-3970` matches the CPU leaf path exactly.
  *This is the fix for finding 10.*
- `launch_partition_split_packed_into` (`:3912-3925`) takes `der1_h` for routing only — either
  handle is behaviourally fine; keep the unsampled one for clarity.
- `launch_der_binary_resident` at `:3978` (next tree's der1) must stay on the unsampled approx —
  unchanged.
- The subtraction trick (`:3861-3869`) operates on whichever histogram was filled; it stays valid
  because the score pair is used consistently at every level.

### Component responsibilities

- `cb-train` owns: RNG stream, sampler choice, sample vector, draw-phase replay, MVS λ carry.
- `cb-compute` owns: the seam type, the plain-host config.
- `cb-backend` owns: uploading the sample, the two elementwise folds, and the score/leaf channel split.

### Error, security, and failure behaviour

- Poisson never reaches the device (gate excludes it); the CPU path continues to return
  `CbError::Degenerate` (`bootstrap.rs:435-438`).
- A length mismatch on `sample` must surface `CbError::LengthMismatch` at the session boundary,
  mirroring the existing `approx` check (`gpu_backend.rs:321-327`).
- `Atomic<u64>`-absent devices already surface `CbError::Unsupported` from the resident histogram;
  do not add a second capability probe.
- `D-10-01` all-or-nothing stays: `begin` returning `Ok(true)` then `grow` returning `Ok(None)` is
  a typed error (`boosting.rs:3200-3212`). Unchanged.
- No secrets, no I/O, no new unsafe.

### Rejected / deferred alternative — Design B (device-resident sampling)

**What it would require.**

| Item | Detail |
|---|---|
| Files | `boosting.rs` (gate + config threading incl. `mvs_lambda`, `rng_seed`), `runtime.rs` (per-tree λ + per-tree RNG base in the seam), `gpu_backend.rs`, `session.rs` (`BootstrapState`/`MvsState` rework), `gpu_runtime/mod.rs` (**the same A4 score/leaf split**) |
| Seam signature | `grow_tree_on_device(approx, target, rng_base: [u64;4], rand_seed: u64, mvs_lambda: Option<f64>)` — the per-tree λ is unavoidable because `DeviceTrainConfig.mvs_lambda` is pinned once at `begin` (`runtime.rs:1096`, read at `session.rs:1592`) while upstream λ changes per tree |
| RNG | The session's own `TFastRng64` seeded from `config.rng_seed` (`session.rs:196-205`, `:1303-1312`) is a **second stream**; it must be deleted and replaced by host-supplied `raw_state()` snapshots, or the host must drive every advance. Even then the host must still replay the per-level/post-tree draws (data-dependent `std_normal`), so Design B does **not** avoid the host draw-replay work — only the per-tree upload |
| Reusable device kernels | Yes: `launch_bootstrap_weights_resident`, `launch_mvs_weights_resident`, `fold_weights_resident`. Poisson's arbitrary stream advance (`session.rs:1567-1577`) would have to be removed or made consumption-faithful |
| Parity risk | **Measured on local gfx1151** [VERIFIED, 2026-07-30]: Bernoulli keep-mask **bit-for-bit** identical to the frozen CPU sample; MVS weights `max_div ∈ {4.441e-16, 6.661e-16, 4.441e-15}` and the per-block threshold `max_div = 5.329e-15`; Bayesian weights `max_div ∈ {4.768e-7, 2.384e-7}`. So Bernoulli/MVS carry essentially no sampler risk; **Bayesian's ~5e-7 weight error is the only real exposure**, and it enters the split-score histogram directly, so it can flip a near-tie border and blow the ≤1e-5 end-to-end bar in a way that is dataset-dependent and not bounded by the weight error |
| New oracles required | Everything Design A needs, **plus** a per-tree RNG-phase equality test (device stream vs host stream), plus a per-tree λ test, plus a Bayesian device-vs-CPU-sample test tightened from ε=1e-4 to ≤1e-5 |

**Verdict: choose Design A now.** Rationale: (a) Design A reuses the *just-corrected, ground-truth
backed* sampler and RNG model verbatim — zero RNG semantics change, which is where every prior
divergence in this area came from; (b) both designs require the identical A4 device rework, so A is
strictly a subset of B's work; (c) B's only advantage is one `8·n`-byte upload per tree, which is a
perf, not a correctness, concern and can be added later behind the same oracle; (d) B re-introduces
the two-stream hazard, per-tree λ plumbing, and a Bayesian numeric exposure with no parity upside.
Design B (or the cleaner "B′" variant where the host snapshots `raw_state()` and advances by the
exact consumed count) is the recommended **follow-up perf phase** once A's oracle is green.

---

## Project Impact Scope

### Must change

| Target | Why | Downstream effects |
|---|---|---|
| `crates/cb-train/src/boosting.rs:3073` (`device_host_eligible`) | the gate under test | opens the device path for 3 bootstrap types |
| `crates/cb-train/src/boosting.rs:3133-3160` (`device_config` + WR-01 comment) | thread `params`; the WR-01 note becomes stale | `session.rs` gate arms observe non-default config (or an explicit host-sampling flag) |
| `crates/cb-train/src/boosting.rs:3196-3486` (device branch) | must consume RNG draws, produce `s`, carry `prev_leaf_mean_l2` | RNG phase for the next tree; MVS λ |
| `crates/cb-compute/src/runtime.rs:1279-1286` (`grow_tree_on_device`) | per-tree sample argument | 3 impl sites |
| `crates/cb-backend/src/gpu_backend.rs:310-332` | forward the sample | — |
| `crates/cb-backend/src/gpu_runtime/session.rs:1449-1710` (`grow_one`) | upload sample, produce score channels | `BootstrapState`/`MvsState` become explicitly unreached-from-`train()` |
| `crates/cb-backend/src/gpu_runtime/mod.rs:3757-4009` (`grow_oblivious_tree_resident`) | **score vs leaf channel split** (finding 10) | affects the `No`-bootstrap path too — must be proven byte-identical when `sample` is empty |
| `crates/cb-train/tests/device_seam_test.rs:69` (`FakeRuntime`) | seam signature | test-only |
| New `crates/cb-train/tests/device_bootstrap_fit_test.rs` | the parity oracle | — |

### May change

| Target | Why |
|---|---|
| `crates/cb-compute/src/runtime.rs:1083-1141` (`DeviceTrainConfig` / `is_covered_regime`) | if an explicit `sample_from_host` flag is added |
| `crates/cb-backend/src/gpu_runtime/session.rs:1043-1091` (gate arms) | if the config now carries a non-default `bootstrap_type` from `train()`, the covered-regime booleans need an arm for "host-sampled" |
| `crates/cb-oracle/generator/gen_fixtures.py:705-830` (`gen_bootstrap`) | if new `boost_from_average=False` bootstrap fixtures are generated (see Testing) |
| `crates/cb-train/src/boosting.rs:3092` (CR-01, `bias == 0.0`) | only if the phase chooses to reuse the existing fixtures directly |
| `bench/bootstrap_gpu/bootstrap_bench.py:112-120` (untracked) | its `only_No_is_gpu_eligible` caveat becomes false |

### Verification only

| Target | Why |
|---|---|
| `crates/cb-train/tests/bootstrap_oracle_test.rs` | CPU no-regression, all 4 scenarios |
| `crates/cb-train/tests/regularization_oracle_test.rs` | `bagging_temp` / `random_strength` scenarios ride the same draw model |
| `crates/cb-train/tests/multidim_sampling_regression_test.rs` | CR-02 per-object sampling invariant |
| `crates/cb-backend/src/kernels/bootstrap_device_test.rs`, `mvs_device_test.rs` | device sampler self-oracles must stay green even though unreached from `train()` |
| `crates/cb-backend/src/gpu_runtime/session_depth_gt1_test.rs:411`, `:628` | gate + finite-tree tests |
| `crates/cb-train/tests/device_nonsym_fit_test.rs`, `device_region_fit_test.rs` | must stay green through the seam change |
| `crates/catboost-rs/src/builder.rs:378-392` | the facade picks `GpuBackend` at compile time; more fits now take the device path |

### Explicitly out of scope

`random_strength != 0` on device; weighted pools (WR-03); Poisson; non-symmetric/Region/CTR/exact ×
bootstrap; Ordered boosting; multi-dimensional approx.

---

## Do Not Hand-Roll

- **The CPU sampler.** `cb_train::bootstrap::bootstrap` and its `fast_log2f`/`fast_logf`
  (`bootstrap.rs:106-135`) are bit-sensitive upstream transcriptions; the file's own comments warn
  that substituting `f32::log2` breaks parity at ~1e-5.
- **The RNG draw model.** `PRE_TREE_DRAWS` / `POST_TREE_EXTRA_DRAWS` / the per-level RSM +
  `CalcScores` + `SelectBestCandidate` counts are instrumented-ground-truth values from `5a5068a`.
  Reuse the constants and call the real draw primitives.
- **`cb_core::std_normal`** — never approximate its draw count.
- **`fold_weights_resident`** — already a generic elementwise f64 product; do not write a second
  multiply kernel.
- **`cb_compute::calc_average` / `scale_l2_reg`** — the frozen leaf/L2 math.
- **`accumulate_leaf_weights` / `normalize_leaf_values`** — the device fold arm already uses them.
- **The device samplers** (`launch_bootstrap_weights_resident`, `launch_mvs_weights_resident`) —
  keep them; they are the Design B asset and are self-oracled.
- **The gate/skip test patterns** — copy `device_nonsym_fit_test.rs`, don't invent a new one.

---

## Common Pitfalls and Risks

| # | Risk | Trigger | Consequence | Prevention | Verification |
|---|---|---|---|---|---|
| 1 | **Sampled leaves** | wiring the existing device bootstrap arm (`session.rs:1636-1645`) without the A4 split | leaves computed from `(unsampled der1, sampled weight)` instead of `(unsampled der1, unsampled weight)`; both channels wrong vs CPU | implement A4 before relaxing the gate | a leaf-value assert in the new device parity test, plus a unit test that `sample`-empty is byte-identical |
| 2 | **Unsampled der in the score histogram** | forgetting `score_der1 = der1·s` | histogram numerator is `Σ der1`, CPU is `Σ der1·s` | A4 | same |
| 3 | **RNG phase drift** | omitting the per-level / post-tree replay on the device branch | tree ≥ 1 samples differ → different model | replay literally, in order | a test comparing `rng.call_count()` after N device trees vs N CPU trees (`TFastRng64::call_count`, `rng.rs:204`) |
| 4 | **`std_normal` count treated as constant** | computing the replay from a formula | drift that only appears on some seeds | call `std_normal` `n_features` times per level | GROUND_TRUTH.md:60-71 records 8/10/12/14 observed for `n_features = 4` |
| 5 | **MVS λ frozen across trees** | using `DeviceTrainConfig.mvs_lambda` | wrong sample from tree 1 onward | Design A computes λ inside `bootstrap()` from `prev_leaf_mean_l2`; the device branch must set it | MVS scenario in the parity test with ≥ 3 iterations |
| 6 | **Fixtures are not device-eligible** | assuming `bootstrap_oracle_test` will exercise the device path | a "passing" upstream oracle that silently ran on the CPU grower | assert device activation explicitly | fixtures pin `boost_from_average=True` (gen_fixtures.py:795) vs `bias == 0.0` at boosting.rs:3092 |
| 7 | **Silent CPU fallback** | any gate arm declining | test passes without touching the device | assert an observable device signature (e.g. `oblivious_trees` non-empty *and* a device-only invariant), as `device_nonsym_fit_test.rs:136-149` does | run under `--features rocm` and diff against a `CpuRefRuntime` fit |
| 8 | **cpu/rocm feature unification** | `cargo test -p cb-train --features rocm` (no `--no-default-features`) | cpu is unified in; the "device" run is not cpu-free | always `--no-default-features --features rocm` | `crates/cb-train/Cargo.toml` `default = ["cpu"]` |
| 9 | **Blanket rocm test build fails** | `cargo test -p cb-train --no-default-features --features rocm` | 9+ `unresolved import cb_backend::CpuBackend` errors | select the target: `--test device_bootstrap_fit_test` | [VERIFIED: blanket run emits `error[E0432]: unresolved import cb_backend::CpuBackend` for `penalty_oracle_test`, `regularization_oracle_test`, `monotone_oracle_test`, … ; targeted `--test device_nonsym_fit_test --no-run` finished OK in 33.16s] |
| 10 | **ε=1e-4 creep** | copying the device test tolerance | the phase goal (≤1e-5) silently weakened | pin ε=1e-5 in the new test and justify any relaxation | `bootstrap_device_test.rs:25`, `mvs_device_test.rs:26`, `device_nonsym_fit_test.rs:167` all use 1e-4 |
| 11 | **Bayesian device sampler drift (Design B only)** | choosing B for Bayesian | ~5e-7 weight error can flip a near-tie split | Design A avoids it entirely | measured `max_div = 4.768e-7` / `2.384e-7` on gfx1151 |
| 12 | **Poisson enabled on device** | relaxing the gate to "!= No" | device succeeds where CPU errors | gate on `!= Poisson` explicitly | `bootstrap.rs:435-438`; no fixture exists (gen_fixtures.py:718-722) |
| 13 | **Per-tree upload cost** | Design A at large `n` | `8·n` bytes/tree; the round-3 perf work showed a 60 MB plain upload dominated runtime | measure with `CB_GPU_PROF=1`; consider Design B′ later | `bench/bootstrap_gpu/bootstrap_bench.py` harness exists |
| 14 | **Stale doc comments** | trusting `bootstrap_device.rs:156` ("exact `ln`") | wrong conclusion about Bayesian parity | read the kernel body | `bootstrap_device.rs:104-111`, `:230` call `fast_log2f` |

---

## Testing and Verification Strategy

### Unit / self-oracle (cb-backend, `*_test.rs` mounted under `#[cfg(test)]`)

- Extend `crates/cb-backend/src/gpu_runtime/session_depth_gt1_test.rs` (or a new
  `session_host_sample_test.rs` mounted the same way) with:
  - `grow_one` with an **empty** sample is byte-identical to today's no-sample grow (D-04).
  - `grow_one` with `s ≡ 1.0` is byte-identical to `s` empty.
  - `grow_one` with a hand-built `s` reproduces a host-computed reference histogram/leaf pair —
    proving the score/leaf channel split.
- Keep `kernels::bootstrap_device_test` and `kernels::mvs_device_test` green (they are the Design B
  asset and the sampler transcription proof).

### Integration / device-vs-CPU parity (cb-train, `tests/`)

New `crates/cb-train/tests/device_bootstrap_fit_test.rs`, modelled verbatim on
`device_nonsym_fit_test.rs`:

- `#[cfg(any(feature = "rocm", feature = "cuda")) ] mod device { … }` with a local `CpuRefRuntime`
  (trait defaults ⇒ CPU grower) and a `#[cfg(not(...))]` `eprintln!("SKIP …")` else-arm.
- Fixture must be **device-eligible**: `boost_from_average = false` (bias 0), unit weights,
  `random_strength = 0.0`, `grow_policy = SymmetricTree`, `leaf_method = Gradient`, RMSE,
  `iterations ≥ 3` (so the continuous RNG stream and MVS λ carry are exercised).
- One scenario per type: Bernoulli (`subsample = 0.8`), Bayesian (`bagging_temperature = 1.0`),
  MVS (`subsample = 0.8`).
- Assert: identical split `(feature, border)` sequences, and `max |Δpred| ≤ 1e-5` between the
  device fit and the `CpuRefRuntime` fit.
- Assert device activation (non-empty `oblivious_trees` + a device-path-only invariant) so a silent
  CPU fallback cannot pass.
- Add an RNG-phase test: after N iterations the device branch's `rng.call_count()` equals the CPU
  branch's. (`TFastRng64::call_count`, `crates/cb-core/src/rng.rs:204`. If `rng` is not observable
  from outside `train_inner`, gate this via a small internal test in
  `crates/cb-train/src/boosting_*_test.rs` following the project's `#[path]`/sibling-file rule.)

### Upstream ≤1e-5 gate (the real parity bar)

The committed `bootstrap/{bayesian,bernoulli,mvs}` fixtures **cannot** gate the device path today:
they pin `boost_from_average=True` (gen_fixtures.py:795; `bootstrap_oracle_test.rs` sets
`boost_from_average: true`) and `device_host_eligible` requires `bias == 0.0` (boosting.rs:3092).
Two routes, pick one and record the decision:

- **Route 1 (recommended, lower risk):** regenerate a parallel fixture family
  `bootstrap_dev/{bayesian,bernoulli,mvs}` with `boost_from_average=False`, everything else
  identical, via `crates/cb-oracle/generator/gen_fixtures.py`. Requires the `catboost` Python
  package on the generation host. `[ASSUMED: the generator runs in this environment]` — see Open
  Questions.
- **Route 2:** lift CR-01 by seeding the resident approx from `bias` in `GpuTrainSession::begin`.
  Larger blast radius (affects every device fit, not just bootstrap ones) and should be its own
  phase.

### Regression / no-regression

- `cargo test -p cb-train --test bootstrap_oracle_test` (all 4 scenarios) — **blocking**.
- `cargo test -p cb-train --test regularization_oracle_test`.
- `cargo test -p cb-train` (full CPU suite). Note the pre-existing failure
  `monotone_non_symmetric_and_region_are_typed_errors` recorded in `5a5068a`'s commit message
  (493 passed / 1 failed on a clean tree).
- `cargo test -p cb-train --no-default-features --features rocm --test device_nonsym_fit_test`
  and `--test device_region_fit_test` (seam-change no-regression).

### Verified commands

```bash
# device sampler self-oracles (VERIFIED green on local gfx1151, 2026-07-30)
cargo test -p cb-backend --no-default-features --features rocm --lib bootstrap -- --nocapture --test-threads 1
cargo test -p cb-backend --no-default-features --features rocm --lib mvs       -- --nocapture --test-threads 1

# targeted cb-train device test under rocm (VERIFIED it BUILDS; blanket run does NOT)
cargo test -p cb-train --no-default-features --features rocm --test device_nonsym_fit_test

# CPU parity gate
cargo test -p cb-train --test bootstrap_oracle_test
```

Recorded results of the two device self-oracle runs (this session, gfx1151):

```
bootstrap bayesian  seed=17 temp=1   n=64   max_div=4.768e-7
bootstrap bayesian  seed=7  temp=0.5 n=1536 max_div=2.384e-7
bootstrap bernoulli seed=17 rate=0.5 kept device=25 cpu=25 / 37     (bit-for-bit)
bootstrap bernoulli seed=42 rate=0.7 kept device=86 cpu=86 / 128    (bit-for-bit)
bootstrap leaves    device=-0.62965180 cpu=-0.62965180 div=0.000e0
mvs seed=17  rate=0.5 n=48  max_div=4.441e-16   kept dev=25 cpu=25
mvs seed=42  rate=0.7 n=64  max_div=6.661e-16   kept dev=47 cpu=47
mvs seed=2024 rate=0.3 n=200 max_div=4.441e-15  kept dev=54 cpu=54
mvs threshold cpu=4.02701297 max_div=5.329e-15
session_bootstrap_gate_covers_bernoulli_bayesian_poisson ... ok
session_bootstrap_grows_finite_tree ... ok        (ran, not skipped — Atomic<u64> available)
7 passed / 0 failed  (bootstrap)   3 passed / 0 failed  (mvs)
```

---

## Planning Guidance

### Suggested work boundaries and ordering

1. **Task 1 — device score/leaf channel split (A4 + A3), `No`-bootstrap behaviour unchanged.**
   Land `grow_oblivious_tree_resident`'s four-handle form and the session's sample upload with an
   *empty* sample everywhere. Prove byte-identity on the existing device tests. This is the largest
   and riskiest change and it is independently verifiable.
2. **Task 2 — seam widening (A2)** across the 3 `Runtime` impls + the `FakeRuntime`.
3. **Task 3 — host draw replay + `prev_leaf_mean_l2` on the device branch (A1 steps 6-8)**, still
   with `bootstrap_type = No` (so the replay is inert and the CPU suite must stay byte-identical).
   Add the `call_count()` equality test here.
4. **Task 4 — relax the gate + thread the config (A1 steps 1-5)** for Bernoulli first (bit-for-bit
   sampler, simplest control mask), then Bayesian, then MVS (needs the λ carry from Task 3).
5. **Task 5 — parity oracles**: device-vs-CPU ≤1e-5 test, then the upstream fixture route decision.
6. **Task 6 (optional, separate phase) — Design B′ perf**: move the draw on-device using the
   already-validated kernels, host-snapshotting `raw_state()` and advancing by the exact consumed
   count.

### Dependencies between tasks

- Task 4 depends on 1, 2, 3. Task 5 depends on 4. Task 6 depends on 5 (its oracle is the gate).
- Task 3's `call_count()` test is the cheapest early warning for the whole phase; front-load it.

### Decisions the planner must preserve

- The host RNG draw model from `5a5068a` is authoritative; do not re-derive.
- CPU sampling semantics: sample affects **only** the split-search histogram
  (`boosting.rs:3723-3731`).
- `sum_all_weights` / `scaled_l2` are tree-invariant and unaffected by sampling
  (`boosting.rs:2376`, `:3802`, device `:3104`).
- Per-fit all-or-nothing (D-10-01) and "never a fabricated device result" (D-04 / T-10-05).
- `cb-backend` must not depend on `cb-train`; `cb-compute` must not see `cubecl` types.
- Source/test separation; ε must be 1e-5 for the sign-off test.

### Items needing a spike or user decision

- **Fixture route decision** (Route 1 regenerate vs Route 2 lift CR-01) — blocks the upstream ≤1e-5
  gate, not the device-vs-CPU gate.
- **Explicit `sample_from_host` flag** on `DeviceTrainConfig` vs implying it — small API decision
  with real readability consequences.
- **Per-tree upload cost spike** at `n = 300k` using `CB_GPU_PROF=1` and the existing
  `bench/bootstrap_gpu/bootstrap_bench.py` harness, to decide whether Design B′ is needed at all.

---

## Open Questions

1. **Which fixture route?** Regenerating `boost_from_average=False` bootstrap fixtures requires the
   `catboost` Python package and a deterministic single-thread run on the generation host; this was
   not exercised in this session `[ASSUMED]`. Without it, the phase can only prove device==CPU, not
   device==upstream.
2. **Is `rng.call_count()` observable from a test?** `train_inner`'s `rng` is local
   (`boosting.rs:3019`). Verifying draw-phase equality may require a small internal test seam or an
   indirect proof (identical models across ≥ 3 iterations). Not resolved here.
3. **Does the ≤1e-5 device-vs-CPU bar actually hold for the existing `No` path?** All shipped
   device e2e tests use ε=1e-4 (`device_nonsym_fit_test.rs:167`). Whether the oblivious device
   grower reproduces the CPU grower to 1e-5 on a non-trivial fixture is **unverified** and is a
   prerequisite for the phase goal, independent of bootstrap. Recommend measuring this first.
4. **Poisson**: confirmed out of scope here; is there any product requirement for GPU-only Poisson
   (upstream supports it on GPU)? If yes it needs its own non-parity acceptance criterion.
5. **Should `BootstrapState`/`MvsState` be deleted or retained** under Design A? Retaining them
   keeps Design B cheap but leaves two samplers in the tree. Recommend retain + comment; needs a
   decision.
6. Context7 was not invoked (no external API in the change surface). If the planner disagrees,
   resolve `cubecl` docs before touching kernel code.

---

## Sources

**Project documents (read directly; TreeFinder MCP was available but the repository's planning
docs are plain files under `.planning/`, read via Read/Bash):**
- `CLAUDE.md` (constraints, source/test separation, CubeCL rules)
- `.planning/plans/bayesian-rng-draw-accounting/instrumented-ground-truth/GROUND_TRUTH.md`
- `.planning/plans/` directory listing (13 prior plan folders)
- `bench/bootstrap_gpu/bootstrap_bench.py` (untracked; records the current "only No is GPU-eligible" caveat)

**CodeGraph queries (`codegraph_explore`):**
- `device_host_eligible DeviceTrainConfig begin_device_training grow_tree_on_device bootstrap`
- `grow_oblivious_tree_resident histogram channels calc_average leaf values weight der1`
- `map_bootstrap_kernel map_mvs_coverage GpuTrainSession::open bootstrap gate DeviceBootstrapKind`
- `bootstrap SampledWeights set_sampled_control bayesian bernoulli mvs_sample_weights EBootstrapType bootstrap.rs`

**Key symbols / paths:**
`crates/cb-train/src/boosting.rs:58,68,2376,3019-3021,3054-3098,3104,3133-3160,3196-3212,3319-3400,3486,3640-3645,3659-3671,3699-3770,3802-3833,4580-4589` ·
`crates/cb-train/src/bootstrap.rs:61-64,106-152,176-188,206-300,348-369,383-439` ·
`crates/cb-train/src/tree.rs:533-546,574-662,1043-1149` ·
`crates/cb-compute/src/runtime.rs:1020-1141,1230-1299` ·
`crates/cb-compute/src/leaf.rs:83-116` ·
`crates/cb-backend/src/lib.rs:33-51` ·
`crates/cb-backend/src/gpu_backend.rs:250,310-341` ·
`crates/cb-backend/src/gpu_runtime/session.rs:86-106,189-225,320-356,704-709,860,1043-1091,1145-1152,1303-1322,1449-1710,1553-1612,1636-1674` ·
`crates/cb-backend/src/gpu_runtime/mod.rs:2083-2103,3757-4009` ·
`crates/cb-backend/src/kernels.rs:2935-2946` ·
`crates/cb-backend/src/kernels/bootstrap_device.rs:51-73,101-150,152-238,240-294,402-486` ·
`crates/cb-backend/src/kernels/mvs_device.rs:60-67,104-258` ·
`crates/cb-core/src/rng.rs:120-265` · `crates/cb-core/src/normal.rs:50` · `crates/cb-core/src/lib.rs:24` ·
`crates/catboost-rs/src/builder.rs:378-392`

**Tests inspected:**
`crates/cb-train/tests/device_nonsym_fit_test.rs` (whole file) ·
`crates/cb-train/tests/device_seam_test.rs` (structure) ·
`crates/cb-train/tests/bootstrap_oracle_test.rs:1-90` ·
`crates/cb-train/tests/multidim_sampling_regression_test.rs:1-60` ·
`crates/cb-backend/src/kernels/bootstrap_device_test.rs` (structure + tolerances) ·
`crates/cb-backend/src/kernels/mvs_device_test.rs` (structure + tolerances) ·
`crates/cb-backend/src/gpu_runtime/session_depth_gt1_test.rs:380-470,600-700`

**Manifests / lockfiles:** `Cargo.toml` (workspace lints + pinned deps) ·
`crates/cb-train/Cargo.toml` · `crates/cb-backend/Cargo.toml`

**Command output (this session, local machine):**
- `rocminfo` → `gfx1151`, AMD Radeon 860M
- `cargo test -p cb-backend --no-default-features --features rocm --lib bootstrap` → 7 passed / 0 failed
- `cargo test -p cb-backend --no-default-features --features rocm --lib mvs` → 3 passed / 0 failed
- `cargo test -p cb-train --no-default-features --features rocm --test device_nonsym_fit_test --no-run` → OK (33.16 s)
- `cargo test -p cb-train --no-default-features --features rocm --no-run` → `error[E0432]: unresolved import cb_backend::CpuBackend` × 9+ targets
- `grep -rln "use cb_backend::CpuBackend" crates/cb-train/tests/ | wc -l` → 37
- `git show --stat 5a5068a`, `git status --short`

**Context7 CLI:** not invoked (no external library API in the change surface). See Open Question 6.
**Web:** not used; every claim is backed by repository evidence or local command output.

---

## Confidence Assessment

### HIGH (directly verified in this session)

1. `device_host_eligible` excludes bootstrap at `boosting.rs:3073-3074` [PROJECT].
2. `DeviceTrainConfig` sampling knobs are left at default with the WR-01 note at
   `boosting.rs:3143-3159` [PROJECT].
3. Device sampling apparatus exists and is self-oracled: `BootstrapState` (`session.rs:196-205`),
   `MvsState` (`session.rs:216-225`), `launch_bootstrap_weights_resident`,
   `launch_mvs_weights_resident` [CODEGRAPH + VERIFIED test run].
4. Two divergent RNG streams: device stream seeded from `config.rng_seed`
   (`session.rs:1303-1322`), advanced only by bootstrap draws (`session.rs:1553-1612`); host stream
   seeded from `params.random_seed` (`boosting.rs:3019`) with the `5a5068a` draw model [PROJECT].
5. Existing device bootstrap e2e test asserts finiteness/determinism only, never parity
   (`session_depth_gt1_test.rs:628-676`) [PROJECT].
6. Poisson's device stream advance is deterministic-but-arbitrary (`session.rs:1567-1577`);
   upstream/CPU reject Poisson (`bootstrap.rs:435-438`, gen_fixtures.py:718-722) [PROJECT].
7. MVS opens the session iff `config.mvs_lambda.is_some()` (`session.rs:1048-1050`, `:1070-1074`);
   `DeviceTrainConfig::default()` has `mvs_lambda: None` (`runtime.rs:1117`) → declines. Pinning it
   opens it [PROJECT].
8. CPU sampling semantics: sample → split histogram only; leaves use unsampled der/weight
   (`boosting.rs:3723-3770`) [PROJECT].
9. `sum_all_weights` (`boosting.rs:2376`) and hence `scaled_l2` are tree-invariant and unaffected by
   sampling; `device_scaled_l2` (`:3104`) matches the CPU per-tree value (`:3802`) [PROJECT].
10. Device seam takes no per-tree weights (`runtime.rs:1279-1286`) [PROJECT].
11. **Finding 10 confirmed:** `grow_oblivious_tree_resident` feeds the SAME `(der1_h, weight_h)`
    into `launch_partition_hist2_resident_into` (`mod.rs:3836-3852`) and
    `launch_partition_update_into` (`mod.rs:3942-3951`), and computes leaves via
    `calc_average(part_stats[3l], part_stats[3l+1], scaled_l2)` (`mod.rs:3963-3970`). The session
    folds the sample into the single `weight_ref` (`session.rs:1636-1645`). Under sampling this is
    wrong on **both** channels [PROJECT].
12. The device branch `continue`s at `boosting.rs:3486` consuming zero RNG and never updating
    `prev_leaf_mean_l2` [PROJECT].
13. `fold_weights_resident` is a generic elementwise f64 product via `vector_mul_kernel`
    (`bootstrap_device.rs:417-439`) — reusable for `der1·s`; no new kernel needed [PROJECT].
14. Bootstrap fixtures exist for `{no,bayesian,bernoulli,mvs}` and all pin `boost_from_average=True`
    (gen_fixtures.py:784-797; `bootstrap_oracle_test.rs`) → excluded by CR-01
    (`boosting.rs:3092`) [VERIFIED].
15. `cargo test -p cb-train --no-default-features --features rocm` fails with
    `unresolved import cb_backend::CpuBackend`; a `--test <target>` run builds fine
    [VERIFIED: both runs].
16. Local device measurements (gfx1151): Bernoulli bit-for-bit; MVS ≤ 6.7e-16 (threshold 5.3e-15);
    Bayesian ≤ 4.77e-7; `session_bootstrap_grows_finite_tree` runs (Atomic<u64> present)
    [VERIFIED: test output].
17. Only 3 `Runtime::grow_tree_on_device` impls exist [VERIFIED: grep].
18. The device Bayesian kernel uses a `#[cube]` `fast_log2f`, not exact `ln` — the module doc is
    stale (`bootstrap_device.rs:104-111`, `:230` vs `:156-157`) [PROJECT].

### MEDIUM (well-supported, not exercised end-to-end here)

19. Design A's draw-replay block, placed after the device grow, yields the identical next-tree RNG
    position as the CPU's interleaved draws. Follows directly from the device grow consuming zero
    draws, but not yet executed.
20. `s[i] = control[i] ? sample_weights[i] : 0.0` is the complete multiplier for both the der and
    weight channels **at unit weight** — derived from `boosting.rs:3738-3770` with
    `eff_weights == weights == 1.0`.
21. Design B′ (host-snapshotted `raw_state()` + exact-count advance) is feasible for Bernoulli
    (exactly `n` draws, `session.rs:1567-1577` already notes this is draw-faithful) and Bayesian
    (1 draw) and MVS (3 draws). Inferred from the sampler code, not implemented.
22. Regenerating `boost_from_average=False` bootstrap fixtures via
    `crates/cb-oracle/generator/gen_fixtures.py` is straightforward — the generator already
    parameterizes `boost_from_average` in `shared` (gen_fixtures.py:795).

### LOW (unresolved; must be validated before or during planning)

23. Whether the device oblivious grower meets **≤1e-5** (not ε=1e-4) against the CPU grower on a
    non-trivial fixture even with `bootstrap_type = No`. All shipped device e2e tests use 1e-4.
    This is a prerequisite for the phase goal and is unverified. (Open Question 3.)
24. Whether the `catboost` Python package is available for fixture regeneration in this
    environment. `[ASSUMED]` (Open Question 1.)
25. Whether the ~4.8e-7 Bayesian device sampler error stays below the ≤1e-5 end-to-end bar on a
    real fixture (relevant only if Design B is chosen for Bayesian).
26. Whether `train_inner`'s `rng` can be observed by a test without a new seam (Open Question 2).
