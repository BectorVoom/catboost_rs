---
phase: 12-grow-policy-leaf-method-sampling-categorical-device-coverage
plan: 08
subsystem: gpu-training
tags: [cubecl, rocm, ctr, categorical, ordered-target-statistic, cindex, device-residency, GPUT-10]

# Dependency graph
requires:
  - phase: 12-01
    provides: "DeviceTrainConfig config surface (incl. the DeviceCtrConfig placeholder) + the all-or-nothing coverage gate in GpuTrainSession::begin"
  - phase: 10-gpu-foundations
    provides: "pack_cindex / read_bin bit-packed cindex + the resident GpuTrainSession seam"
provides:
  - "Device ordered / one-hot / tensor CTR accumulation resident across the learn permutation (read-before-increment, no leakage) — matches the CPU online_ctr_prefix_binclf column ≤1e-4 (good/total EXACT), validated in-env on BOTH cpu and rocm gfx1100"
  - "CTR→cindex binarize JOIN: accumulated CTR values binarize into ADDITIONAL cindex columns on device (> bin convention), joined via pack_cindex"
  - "Single-permutation CTR gate arm in begin: covered CTR → device (augments the resident cindex); multi-fold/multi-permutation CTR → Ok(None) (Open Q3)"
  - "DeviceCtrConfig filled (permutation / target_class / per-column priors+projections+borders) + DeviceCtrColumn"
affects: [12-09-coverage-matrix-kaggle-signoff]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Serial #[cube] read-before-increment ordered-prefix scan (unit 0) — the ordered CTR is inherently sequential; exact INTEGER prefix counting needs no Atomic<u64>, so the self-oracle runs in-env on the cpu backend as well as rocm/cuda"
    - "Host tensor/feature-combination projection (combine_projection_bins) folds member categories into a combined bin column feeding the SAME device CTR math (A5)"
    - "CTR VALUES stay on device; only the final integer binarized bin columns are host-packed (the A2 cindex host-pack-once discipline extended to CTR — NOT host-computed-then-uploaded)"
    - "CTR gate arm (Pattern A): covered single-permutation CTR augments the resident cindex during begin; multi-fold declines via the fold_count!=1 gate (Open Q3)"

key-files:
  created:
    - crates/cb-backend/src/kernels/ctr_device.rs
    - crates/cb-backend/src/kernels/ctr_device_test.rs
  modified:
    - crates/cb-backend/src/kernels.rs
    - crates/cb-compute/src/runtime.rs
    - crates/cb-compute/src/lib.rs
    - crates/cb-backend/src/kernels/cindex.rs
    - crates/cb-backend/src/gpu_runtime/session.rs
    - crates/cb-backend/src/gpu_runtime/session_depth_gt1_test.rs

key-decisions:
  - "The ordered binclf CTR prefix is EXACT INTEGER counting (N0/N1 per bucket), NOT a float reduction — so Pattern C's fixed-point Atomic<u64> reduce is NOT required here (it applies to the FLOAT CTR sums, out of this binclf ordered-TS scope). A serial single-thread #[cube] scan (like the bootstrap Bernoulli kernel) is the faithful, deterministic device transcription, and it runs in-env on cpu (no Atomic<u64> dependency) — the CTR self-oracle is fully validatable in-env, on BOTH cpu and rocm gfx1100."
  - "A5 (tensor-CTR kernel sharing): the tensor/feature-combination CTR reuses the SAME ordered-prefix kernel; only the combined-projection pre-step differs (a host fold via combined_hash/fold_cat_hash/calc_hash transcribed inline). One-hot likewise shares the kernel (bucket key = raw small-cardinality category). No separate kernel per CTR flavour."
  - "A6 (cat-heavy fixture): the Phase-10 float-ramp generator has no categorical features, so the CTR self-oracle uses local deterministic LCG categorical + binclf-class + permutation fixtures (synth_fixture) — no new shared fixture crate needed."
  - "Open Q3 (single-permutation first): the CTR gate covers ONLY the single-permutation regime (fold_count==1). A multi-fold / multi-permutation CTR is declined by the existing fold_count!=1 gate → Ok(None); multi-permutation CTR defers to a later wave. The gate is explicitly asserted (fold_count=2 CTR → None)."
  - "The covered CTR path augments the resident cindex ON device during begin (accumulate → binarize → append columns, CTR values never touch the host) and returns Ok(Some); the full-tree grow numerics over the augmented cindex + the exact upstream combined_hash parity are the Plan-09 Kaggle CUDA sign-off. GPUT-10 stays Pending — not fabricated (Phase-11-05 precedent: never fabricate a GPU oracle)."

patterns-established:
  - "wgpu f64 CTR path rejected with a typed CbError (WR-02) — the CTR value seam is f64/u64; the in-env cpu/rocm/cuda path is unaffected. CTR tests SKIP off the f64 backends (WR-01 anti-false-pass)."

requirements-completed: []  # GPUT-10 stays Pending — device CTR accumulation + join + gate landed & self-oracled ≤1e-4, but the authoritative full-tree Kaggle CUDA sign-off is Plan 09.

# Metrics
duration: ~40min
completed: 2026-07-04
status: complete
---

# Phase 12 Plan 08: Device CTR / Permutation-Dependent Categorical Features Summary

**Ported the upstream device CTR computation so ordered / one-hot / tensor target-statistic CTRs accumulate ON device, resident across the learn permutation (read-before-increment, no leakage), and binarize into ADDITIONAL cindex columns on device — matching the CPU `online_ctr_prefix_binclf` column ≤1e-4 (good/total EXACT), validated in-env on BOTH the cpu backend AND real rocm gfx1100; a single-permutation CTR gate arm covers the regime (multi-fold defers behind `Ok(None)`), with the authoritative full-tree sign-off deferred to Plan 09.**

## Performance

- **Duration:** ~40 min
- **Completed:** 2026-07-04
- **Tasks:** 2
- **Files:** 2 created, 6 modified

## Accomplishments

- **Device ordered CTR (Task 1):** `ctr_device.rs` — a serial `#[cube]` read-before-increment ordered-prefix kernel (port of `online_ctr.cpp:300-307` `CalcQuantizedCtrs`, transcribed inline, NO `cb-train` dep) accumulates the per-bucket `[N0, N1]` prefix resident across the permutation and emits per-object `(good, total, value)` in object order. The FIRST document in each bucket reads the PRIOR alone (no leakage). Exact integer counting → no `Atomic<u64>` needed → the serial scan runs in-env on cpu.
- **Tensor / one-hot (A5):** `combine_projection_bins` folds member categories into one combined-projection bin column (`combined_hash`/`fold_cat_hash`/`calc_hash` transcribed) feeding the SAME prefix kernel; one-hot shares the kernel with the raw category as the bucket key. No per-flavour kernel.
- **CTR→cindex JOIN (Task 2):** `binarize_ctr_kernel` binarizes accumulated CTR VALUES into cindex bins on device (`bin = #{borders < value}`, the `> bin` convention every cindex consumer reads); the binarized column packs into the cindex as an ADDITIONAL feature via `pack_cindex`.
- **CTR gate arm (Task 2):** `ctr_covered` + `build_ctr_cindex_columns` in `session.rs`. A covered single-permutation CTR config accumulates + binarizes the extra columns ON device during `begin` (CTR values never touch the host — the A2 host-pack-once discipline), augments the resident cindex, and returns `Ok(Some)`; a multi-fold/multi-permutation CTR declines to `Ok(None)` (Open Q3). Added `n_features_effective()`.
- **Self-oracle (Task 2):** `ctr_device_test.rs` — device ordered/one-hot/tensor CTR vs an inline serial CPU reference (transcribed `online_ctr_prefix_binclf`/`calc_ctr_online`, no `cb-train` dep): good/total EXACT, value ≤1e-4; first-doc-reads-prior; bit-exact CTR→cindex round-trip. Plus the `cindex.rs` JOIN test and the `session` gate + resident-augment tests.

## A5 / A6 / Open-Q3 findings (required by plan output)

- **A5 (tensor-CTR kernel sharing):** confirmed — the tensor/feature-combination CTR reuses the SAME ordered-prefix device kernel; only the combined-projection host fold differs. One CTR kernel serves ordered TS, one-hot, and tensor combos.
- **A6 (cat-heavy fixture):** the Phase-10 generator lacks categorical features, so local deterministic LCG categorical/class/permutation fixtures were added in the test module (no new shared fixture crate).
- **Open Q3 (single-permutation first):** the CTR gate covers ONLY `fold_count==1`; multi-fold/multi-permutation CTR is declined to `Ok(None)` (asserted `fold_count=2` CTR → None) and deferred to a later wave.

## Verification

- `cargo test -p cb-backend ctr` — **cpu (default):** 8/8 green; **rocm gfx1100 in-env:** 8/8 green (device ordered/one-hot/tensor CTR ≤1e-4 vs CPU, good/total exact; CTR→cindex bit-exact; gate + resident-augment).
- `cargo test -p cb-backend session_` — 10/10 green (cpu; the Atomic<u64>-dependent grow oracles SKIP per WR-01, the CTR gate + augment arms RUN).
- `cargo build -p cb-backend` + `cargo build -p cb-backend --features rocm --tests` — green (ROCm smoke, `#[cube]` codegen).
- `cargo check -p cb-train --tests` + `cargo build -p cb-compute` — green (no cross-crate break from the `DeviceCtrConfig` field addition).
- Landmine checks: `grep -rn "use cb_train" crates/cb-backend/src/kernels/ctr_device.rs` empty; no `-inf` literal in any `#[cube]` body (finite border comparisons only); source/test separation held (all `#[test]` in `*_test.rs` / the cindex test file).

## Deviations from Plan

### 1. [Rule 3 — Blocking] CTR→cindex JOIN test uses the HOST `read_bin` accessor, not the in-env-broken device `read_all_bins_kernel`

- **Found during:** Task 2 verification.
- **Issue:** The pre-existing GPUT-15 cindex device-read oracle (`pack_read_bit_exact_*`, Plan 10-06) FAILS in-env on the cpu backend — `read_all_bins_kernel`'s grid-stride read-back returns scrambled bins (a cpu-runtime grid-stride execution defect). Confirmed pre-existing (reproduces on the ORIGINAL `cindex.rs` from HEAD, in isolation). My CTR JOIN test initially reused that path and failed for the SAME environmental reason.
- **Fix:** the JOIN test packs the CTR-augmented matrix and extracts every cell via the deterministic HOST `read_bin_host` (the exact `(word >> shift) & mask` device `read_bin` math), proving the binarized CTR column packs + extracts as an additional cindex feature bit-exact — independent of the broken grid-stride kernel. The device `read_all_bins_kernel` path is the GPUT-15 oracle's own (Kaggle) concern.
- **Files modified:** `crates/cb-backend/src/kernels/cindex.rs`.
- **Logged:** `deferred-items.md` (Plan 12-08, SCOPE BOUNDARY — pre-existing, GPUT-15 not GPUT-10).

## Deferred Issues

- Pre-existing GPUT-15 `read_all_bins_kernel` cpu-runtime grid-stride read-back defect (all `pack_read_bit_exact_*` fail in-env; pre-existing, unrelated file). Logged in `deferred-items.md`.
- Pre-existing `nonsym_grow.rs` / `nonsym_grow_test.rs` `mut`-not-needed warnings (Plan 03 files, out of scope). My changed files are warning-clean.

## Known Stubs

- None. `DeviceCtrConfig` is now filled (Plan-01's documented placeholder resolved). The full-tree grow over the CTR-augmented cindex is NOT a stub — it is a scoped deferral to the Plan-09 Kaggle CUDA sign-off (GPUT-10 stays Pending, not fabricated), consistent with the all-or-nothing coverage gate and the environment ROCm Atomic<u64> regression.

## Self-Check: PASSED

- Files exist: `ctr_device.rs`, `ctr_device_test.rs` ✓
- Commits exist: `f8c4651` (Task 1), `7124fc3` (Task 2) ✓
- Device CTR self-oracle ≤1e-4 (good/total exact) on cpu AND rocm gfx1100 — 8/8 CTR tests green ✓
- `grep "use cb_train" ctr_device.rs` empty ✓; multi-fold CTR → `Ok(None)`, single-perm covered CTR → `Ok(Some)` asserted ✓
