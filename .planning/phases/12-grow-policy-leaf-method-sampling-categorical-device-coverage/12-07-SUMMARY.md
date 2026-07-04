---
phase: 12-grow-policy-leaf-method-sampling-categorical-device-coverage
plan: 07
subsystem: gpu-training
tags: [cubecl, rocm, mvs, sampling, minimal-variance-sampling, threshold, tfastrng64, coverage-gate, GPUT-17]

# Dependency graph
requires:
  - phase: 12-grow-policy-leaf-method-sampling-categorical-device-coverage
    plan: 01
    provides: "DeviceTrainConfig (bootstrap_type=Mvs / sample_rate / mvs_lambda / rng_seed) + all-or-nothing coverage gate; the resident GpuTrainSession"
  - phase: 12-grow-policy-leaf-method-sampling-categorical-device-coverage
    plan: 06
    provides: "map_bootstrap_kernel gate arm template + BootstrapState pattern; fold_weights_resident (device weight * sample); the serial #[cube] TFastRng64 per-block reseed transcription (NextUniformF)"
  - phase: 11-depth6-grow
    provides: "resident der seam (device-resident der1 over SelectedRuntime) the MVS kernel reduces over"
provides:
  - "Device MVS sample-weight draw (per-block threshold over sqrt(lambda+der^2) + inverse-probability reweight) reproducing cb-train mvs_sample_weights ≤1e-4, kept device-resident (D-08)"
  - "map_bootstrap_kernel MVS arm: bootstrap_type==Mvs flips Ok(None)->device (when mvs_lambda pinned); MvsState + grow_one MVS arm folds the resident sample into the resident weight"
  - "launch_mvs_weights_resident (resident Handle) + draw_mvs_weights_host (readback oracle wrapper)"
affects: [12-09-coverage-matrix]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Serial single-thread #[cube] MVS kernel (unit 0 loops blocks) — deterministic block sums by construction (no Atomic<f64>/Atomic<u64>), so it RUNS in-env on gfx1100 despite the Atomic<u64>-advertisement regression that blocks the resident histogram (like Plan 06 RNG / Plan 08 CTR)"
    - "Threshold = deterministic monotone BISECTION of the calculate_threshold root (F(mu)=Sigma min(1,c/mu)=sample_size is continuous strictly-decreasing -> unique root); 'match the threshold SEMANTICS, not the algorithm' (CUDA design §6.1) — 100 halvings match the CPU quickselect root to ~5e-15"
    - "Conditional NextUniformF draw (p > f64::EPSILON) keeps the per-block RNG phase bit-aligned with the CPU stream; sample_rate>=1 short-circuits to all-1.0 with ZERO draws"

key-files:
  created:
    - crates/cb-backend/src/kernels/mvs_device.rs
    - crates/cb-backend/src/kernels/mvs_device_test.rs
  modified:
    - crates/cb-backend/src/kernels.rs
    - crates/cb-backend/src/gpu_runtime/session.rs

key-decisions:
  - "Threshold via deterministic monotone bisection instead of literally invoking the Plan-05 segmented sort primitive: the CPU reference calculate_threshold is a recursive quickselect whose result is the UNIQUE root of F(mu)=sample_size; a 100-iter bisection finds that same root to ~5e-15 (measured), and the plan + CUDA design §6.1 explicitly sanction 'match the threshold SEMANTICS, not the algorithm'. This avoids composing a multi-kernel device sort (which reads back to host) inside the per-tree MVS — keeping the whole reduction device-resident (D-08) and JIT-simple. The mvs_device->sort key_link is satisfied SEMANTICALLY (the block ordering the sort would establish is exactly what the monotone root-find encodes), not by a literal sort.rs call."
  - "Serial single-thread kernel (unit 0), mirroring Plan 06/08: every block SUM is an order-fixed serial accumulation — inherently deterministic (Pattern 3 satisfied WITHOUT any device atomic), and crucially it uses NO u64 atomic, so the device MVS oracle RUNS in-env on gfx1100 (3/3 green) while the resident-histogram grow oracles stay red on the Atomic<u64>-advertisement regression."
  - "MVS coverage requires the caller to pin config.mvs_lambda (GetLambda supplied by the CPU harness): the iter-0 auto-lambda = mean(|der|)^2 from the RESIDENT derivatives needs a deterministic device der reduction, deferred (MVP scope). grow_one keeps a defensive host-der fallback (mvs_lambda_from_der over host_der1) but the gate declines an unpinned MVS to CPU."
  - "MvsState is distinct from BootstrapState (MVS is a derivative REDUCTION, not the RNG-only draw); BootstrapArm gains an Mvs variant (replacing the Plan-06 Decline). The two states are mutually exclusive (distinct bootstrap_type)."

patterns-established:
  - "Device threshold estimators can match a recursive-quickselect CPU reference to full f64 precision via a monotone root-find when the estimator is the root of a continuous monotone equation — no sort, no recursion, no atomics, HIP-JIT-simple."

requirements-completed: []  # GPUT-17 device MVS kernel + gate landed + self-oracled ≤1e-4 in-env; the authoritative Kaggle CUDA epsilon sign-off is the Plan-09 gate (left Pending here).

# Metrics
duration: ~45min
completed: 2026-07-04
status: complete
---

# Phase 12 Plan 07: Device Minimal-Variance Sampling (MVS) Summary

**Minimal-Variance Sampling — CatBoost's DEFAULT GPU sampler — now runs ON device: a serial `#[cube]` kernel finds each block's optimal threshold over `sqrt(lambda+der^2)` by a deterministic monotone bisection of the same `calculate_threshold` root, then inverse-probability-reweights each object via the CPU's exact per-block-reseeded `NextUniformF` stream — reproducing `cb-train::mvs_sample_weights` to ~5e-15 (kept-counts bit-exact) on gfx1100 in-env, kept device-resident (D-08), behind an all-or-nothing MVS gate arm.**

## Performance

- **Duration:** ~45 min
- **Completed:** 2026-07-04
- **Tasks:** 2
- **Files:** 2 created, 2 modified

## Accomplishments

- **Device MVS kernel (Task 1) — `mvs_device.rs`:** a serial (unit 0) `#[cube]` kernel over the resident derivatives. Per block (`BlockSize = 8192`):
  1. **candidate** `c_i = sqrt(lambda + der_i^2)`;
  2. **threshold** — the block `mu` solving `Σ min(1, c_i/mu) = sample_rate·blockSize`, found by a deterministic monotone **bisection** (100 halvings → ~2^-100 relative). This is the UNIQUE root the CPU recursive quickselect `calculate_threshold` returns ("match the threshold SEMANTICS, not the algorithm", CUDA design §6.1). Every block SUM is a serial order-fixed accumulation — deterministic with **no device atomic** (so it dodges the in-env `Atomic<u64>` regression);
  3. **reweight** — `p = single_probability(c_i, mu)`; `weight = 1/p` when `NextUniformF < p` else `0`, over the CPU's exact per-block reseed `from_seed(rand_seed + block_idx).advance(10)` (transcribed inline from `cb_core::TFastRng64`), with the draw CONDITIONAL on `p > f64::EPSILON` (phase-aligned with the CPU). `launch_mvs_weights_resident` returns a resident `Handle` (no read-back); `draw_mvs_weights_host` is the oracle readback wrapper. `wgpu` rejected (WR-02); no `-inf` literal; no `cb-train` dep.
- **Self-oracle + gate arm (Task 2) — `mvs_device_test.rs` + `session.rs`:**
  - The test transcribes `cb-train`'s `single_probability` / `calculate_threshold` / `mvs_sample_weights` INLINE (over the validated `cb_core::TFastRng64` + ordered `sum_f64` — non-tautological, no `cb-train` dep) and asserts the device weights reproduce the frozen CPU sample ≤1e-4, kept-count identical, per-block threshold consistent, multi-block reseed deterministic. Assertions SKIP off rocm/cuda (WR-01 anti-false-pass).
  - `map_bootstrap_kernel` gains `BootstrapArm::Mvs` (replacing Plan-06's `Decline`); the coverage gate adds `mvs_covered` (caller-pinned `mvs_lambda`, every other family flag default — D-10-01 all-or-nothing PER family; an unpinned-λ MVS declines to CPU). `MvsState` holds the continuous stream; `grow_one`'s MVS arm takes the one main-stream `GenRand()` as `rand_seed`, launches the resident MVS draw over `der1_h`, folds it into the resident weight (`fold_weights_resident`, shared with Plan 06), and advances the stream by the `performRandomChoice=false` compensation draws. `sample_rate >= 1.0` short-circuits to all-`1.0` with zero draws (CPU parity).

## Verification

- `cargo test -p cb-backend --features rocm mvs` — **rocm gfx1100 in-env: 3/3 green** (the serial MVS kernel EXECUTES, not skipped):
  - `mvs_weights_match_frozen_cpu_sample_within_epsilon` — max_div **4.4e-16 … 4.4e-15**, kept dev==cpu exact (25/25, 47/47, 54/54).
  - `mvs_per_block_threshold_matches_and_reweight_is_consistent` — cpu_threshold=4.02701297, max_div **5.3e-15**; kept un-capped `weight == threshold/candidate`.
  - `mvs_multi_block_reseed_is_deterministic_and_finite` — n=8216 (2 blocks), pinned-seed determinism, tail block visited (15/24 kept).
- `cargo test -p cb-backend mvs` (cpu) — 3/3 (device assertions skip per WR-01; compilation + gate wiring exercised).
- `cargo test -p cb-backend --features rocm bootstrap` — 7/7 (no Plan-06 regression).
- `cargo build -p cb-backend` + `--features rocm` — green (ROCm smoke; no `-inf`/JIT reject).
- `cargo check -p cb-backend --tests` — green.
- Landmine greps: `use rand|use cb_train` in `mvs_device.rs` → **CLEAN**; `cb-train` dep in `cb-backend/Cargo.toml` → **CLEAN**; no `-inf` literal in kernel body.

## Task Commits

1. **Task 1: Device MVS per-block threshold + inverse-probability reweight** — `e6d64b4` (feat)
2. **Task 2: MVS frozen-sample self-oracle + MVS session gate arm** — `5526c51` (feat)

## Deviations from Plan

### 1. [Rule 3 — Blocking / design] Threshold via monotone bisection, not the literal Plan-05 sort primitive

- **Found during:** Task 1 design.
- **Issue:** The plan's `<action>` and the `mvs_device -> sort.rs` key_link describe finding the threshold via "block sort (Plan-05 primitive) + prefix scan + GetThreshold". But `sort.rs`'s only device primitive is `segmented_radix_sort`, a HOST-orchestrated multi-kernel that READS BACK to host per segment — composing it inside the per-tree MVS would round-trip the block candidates to the host, conflicting with the "runs on device as a reduction over resident derivatives" must-have and the D-08 residency discipline. It would also add a large JIT surface.
- **Fix:** The CPU reference `calculate_threshold` (recursive quickselect) returns the UNIQUE root of the continuous, strictly-decreasing `F(mu) = Σ min(1, c_i/mu) = sample_size`. A deterministic 100-iter monotone bisection finds that same root to ~5e-15 (measured), with only serial order-fixed sums (no atomics, no sort, no recursion). Both the plan (`<action>`: "matching `calculate_threshold` semantics") and `CATBOOST_CUDA_KERNELS_DESIGN.md` §6.1 ("match the threshold SEMANTICS, not the algorithm") explicitly sanction this. The `sort` key_link is satisfied SEMANTICALLY (the monotone root-find encodes exactly the ordering a sort + GetThreshold would establish), not by a literal `sort.rs` call.
- **Files:** `crates/cb-backend/src/kernels/mvs_device.rs`.
- **Commit:** `e6d64b4`.

### 2. [Rule 3 — Blocking] MVS coverage requires a caller-pinned `mvs_lambda`

- **Found during:** Task 2 gate wiring.
- **Issue:** The CPU `mvs_lambda(derivatives, prev_leaf_mean_l2)` derives λ from the (host) derivatives each tree. In the resident session the derivatives are device-resident (`der1_h`); deriving the iter-0 `mean(|der|)^2` on device needs a deterministic device der reduction (out of MVP scope).
- **Fix:** The coverage gate covers MVS only when the caller pins `config.mvs_lambda` (the CPU harness computes `GetLambda` and supplies it) — an unpinned-λ MVS declines to CPU (`Ok(None)`, never a wrong device result). `grow_one` retains a defensive host-der fallback (`mvs_lambda_from_der` over `host_der1`) for robustness. The self-oracle exercises both the iter-0 formula and an explicit later-tree λ directly. Documented as MVP scope; auto-λ-from-resident-der is a follow-up.
- **Files:** `crates/cb-backend/src/gpu_runtime/session.rs`.
- **Commit:** `5526c51`.

## Deferred Issues

- **Kaggle CUDA ε=1e-4 sign-off** for the full MVS-in-boosting run is the Plan-09 gate (per plan) — this plan locks the MVS DRAW numerics by the device-vs-frozen-CPU self-oracle only (in-env rocm ~5e-15). GPUT-17 stays Pending until the Plan-09 CUDA sign-off.
- **Multi-tree continuous-stream parity in the SESSION** advances the host RNG per tree (one `rand_seed` + two `performRandomChoice=false` compensation draws), but is validated end-to-end only by the (env-blocked) resident-grow oracle; the per-draw numerics are locked by the self-oracle.
- **iter-0 auto-λ from the resident derivatives** (device `mean(|der|)^2`) is deferred — the covered device MVS regime uses the caller-pinned `config.mvs_lambda`.

### Pre-existing (out of scope — SCOPE BOUNDARY, already logged in `deferred-items.md`)

- The in-env ROCm `Atomic<u64>`-advertisement regression (`gpu_runtime/mod.rs:1826`) fails the 3 resident-histogram grow oracles (`session_depth_gt1_grows_and_matches_direct`, `session_residency_matches_cpu_multi_tree_boosting`, `session_exact_leaf_grows_finite_quantile_leaves`) with `Unsupported("partition-aware histogram fill requires Atomic<u64> add …")`. Confirmed identical `Unsupported` failure; environment-wide, pre-existing (Plan 12-06), unrelated to MVS. The Plan-07 MVS kernel uses **no** u64 atomic and its 3 oracles pass in-env.

## Known Stubs

- None. The device MVS kernel, gate arm, and resident fold are fully implemented and self-oracled ≤1e-4 (rocm in-env ~5e-15). The only in-env-blocked path (the e2e resident grow) is blocked by the pre-existing `Atomic<u64>` capability state, not by a stub.

## Self-Check: PASSED

- Files exist: `mvs_device.rs`, `mvs_device_test.rs` ✓
- Commits exist: `e6d64b4`, `5526c51` ✓
- `mvs_device.rs` contains `mvs` (kernel + launch); `mvs_device_test.rs` contains `max_divergence`; `session.rs` contains the MVS gate arm (`BootstrapArm::Mvs` / `mvs_covered` / `MvsState`) ✓
- rocm 3/3 MVS oracles green in-env (max_div ≤ 5.3e-15); no `rand`/`cb_train`; no cb-train dep; no `-inf` ✓
