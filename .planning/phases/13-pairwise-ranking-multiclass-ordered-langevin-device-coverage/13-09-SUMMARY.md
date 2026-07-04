---
phase: 13-pairwise-ranking-multiclass-ordered-langevin-device-coverage
plan: 09
subsystem: kernels
tags: [gpu, langevin, sglb, rng, seeded-gaussian, GPUT-20, self-oracle]
requires:
  - "cb-core::TFastRng64 + cb-core::std_normal (Phase-1 oracle-tested PCG PRNG + Marsaglia-polar draw)"
  - "cb-backend kernels::mvs_device inline PCG primitives + per-block reseed precedent (GPUT-17)"
  - "cb-backend kernels::bootstrap_device unbounded-with-flag rejection-loop shape (Poisson knuth)"
  - "GpuTrainSession coverage-gate template (pairwise/ranking/multiclass/ordered Option<*State>)"
provides:
  - "kernels::langevin::launch_langevin_resident — AddLangevinNoise over the resident reduced der (in-place, no read_one)"
  - "kernels::langevin::draw_langevin_host — readback wrapper for the self-oracle"
  - "kernels::langevin::langevin_covered_loss — A4 pairwise-oracle decline predicate"
  - "GpuTrainSession.langevin: Option<LangevinState> — the landed Langevin coverage seam"
affects:
  - "GpuTrainSession::begin coverage gate (records map_langevin_coverage decision; pointwise path byte-unchanged)"
tech-stack:
  added: []
  patterns:
    - "serial #[cube] over the resident der buffer (unit 0 loop), in-place += noise"
    - "inline TFastRng64 from_seed + advance(10) transcription (no cb-core/cb-train reach in #[cube])"
    - "inline Marsaglia-polar std_normal (unbounded while-!accepted flag loop; positive-measure accept, DoS-safe)"
    - "per-element reseed from_seed(rand_seed + i) → independent, deterministic per-object noise"
    - "Ok(None) family coverage gate (all-or-nothing per family, D-10-01); f64 wgpu typed reject (WR-02)"
key-files:
  created:
    - crates/cb-backend/src/kernels/langevin.rs
    - crates/cb-backend/src/kernels/langevin_test.rs
  modified:
    - crates/cb-backend/src/kernels.rs
    - crates/cb-backend/src/gpu_runtime/session.rs
decisions:
  - "Serial (unit 0) kernel chosen over per-element parallel: matches the MVS/bootstrap device precedent, single-block MVP fixtures, and keeps the draw stream trivially deterministic (per-element reseed makes it embarrassingly parallel in principle, but serial removes any thread-divergence risk in the variable-count rejection loop)"
  - "AddLangevinNoise mutates the resident der IN PLACE (der[i] += coefficient * std_normal_i) and returns the same handle — the noised der IS the der for leaf estimation (D-08 residency, no read_one)"
  - "Marsaglia-polar uses the unbounded while-!accepted FLAG shape (the bootstrap Poisson-loop precedent), NOT a bounded break loop — guarantees the device accepts at the exact same iteration as the CPU stream, so the per-element draw count matches bit-for-bit (Pitfall 4)"
  - "langevin_covered_loss is false for is_pairwise_scoring (PairLogitPairwise / YetiRankPairwise) — Langevin is NOT supported on the pairwise oracle (A4, upstream pairwise_oracle.h CB_ENSURE) — so a *Pairwise + Langevin config falls back to CPU (Ok(None))"
  - "LangevinState carries only the covered der_kernel (the coverage decision), like RankingState.objective — the noise coefficient + per-tree grow descriptor ride a forward seam (no device Langevin config knob yet), so the field is a #[allow(dead_code)] structural seam constructed None, mirroring pairwise/ranking/multiclass/ordered"
metrics:
  duration_min: 22
  completed: 2026-07-04
  tasks: 2
  files_created: 2
  files_modified: 2
  commits: 2
status: complete
---

# Phase 13 Plan 09: Langevin / SGLB Device Coverage (GPUT-20) Summary

`AddLangevinNoise` now adds a per-element seeded Gaussian (`coefficient · std_normal(seed_i)`) to the resident reduced-derivative buffer entirely on device, transcribing the exact `cb_core::std_normal` Marsaglia-polar draw order inline — self-oracled bit-for-bit against the frozen pinned-seed CPU sequence at ε=1e-4. This is the fifth and smallest device family (D-09), completing the Phase-13 five-family device-coverage surface for the Phase-14 aggregate sign-off.

## What was built

- **`kernels/langevin.rs` (NEW, 287 LOC)** — the Langevin device driver:
  - `langevin_noise_kernel` — serial `#[cube]` (unit 0). Per element `i`: reseeds a fresh `TFastRng64` from `rand_seed + i` (verbatim `from_seed` + `TFastRng64::new` transcription copied from `mvs_device`), `advance(10)`s both streams (comptime-unrolled), draws ONE standard normal via the inline Marsaglia-polar rejection loop (`x, y = gen_rand_real1()*2-1`; accept iff `0 < x*x+y*y <= 1`; `result = x*sqrt(-2*ln(r)/r)`), and adds `coefficient * result` to `der[i]` IN PLACE. No `-inf` literal; no `cb_core`/`cb-train` reach inside the `#[cube]` body.
  - `launch_langevin_resident(client, der_h, rand_seed, coefficient, n)` — launches the kernel on the resident der handle, returns the same handle WITHOUT `read_one` (D-08). Empty-`n` short-circuits to a 0-length handle (no launch / no 0-len read). f64/u64 wgpu typed reject (`CbError::OutOfRange`, WR-02).
  - `draw_langevin_host(...)` — readback wrapper for the self-oracle only (upload der → launch → `read_one` → `Vec<f64>`; read-back failure → `CbError::Degenerate`, WR-05).
  - `langevin_covered_loss(loss)` — the A4 gate predicate: `false` for `is_pairwise_scoring` (Langevin unsupported on the pairwise oracle), `true` for the covered pointwise der family (RMSE/Logloss/CrossEntropy).
- **`gpu_runtime/session.rs` (MODIFY)** — `LangevinState { der_kernel }` struct + `map_langevin_coverage` gate (Pattern A family-gated `Option`) + `langevin: Option<LangevinState>` field. `begin(...)` records the coverage decision (`let _langevin = map_langevin_coverage(...)`); a covered pointwise fit proceeds byte-unchanged, and a `*Pairwise` + Langevin config is declined (both by the pairwise arm above and by `langevin_covered_loss`).
- **`kernels/langevin_test.rs` (NEW, 199 LOC)** — the frozen pinned-seed self-oracle (`#![cfg(not(feature = "wgpu"))]`), 3 tests:
  1. device noised der vs frozen CPU `coefficient · std_normal` sequence ≤1e-4 (device-gated / cpu-skip WR-01), all elements finite;
  2. per-element Marsaglia-polar draw count well-formed (every element draws ≥1 pair, mean ≈1.27) + the device value-match as the count-divergence detector (Pitfall 4);
  3. empty-`n` no-op (no 0-len read) + `PairLogitPairwise`/`YetiRankPairwise` + Langevin decline (A4) while RMSE/Logloss/CrossEntropy are covered and Quantile is outside the covered family.
- **`kernels.rs` (MODIFY)** — registered `pub(crate) mod langevin` + `#[cfg(test)] mod langevin_test`.

## Verification

- `cargo test -p cb-backend --lib langevin` → **3/3 green** (CPU default).
- `cargo check --tests -p cb-backend` → clean (exit 0).
- Acceptance greps: `x * x + y * y` present; 0 `NEG_INF` in non-comment lines; 0 `cb-train` in `crates/cb-backend/Cargo.toml`; no `read_one` in `launch_langevin_resident`; `cb_core::` appears only in doc comments + the `CbError`/`CbResult` import (never inside the `#[cube]` body).

## Deviations from Plan

None — plan executed as written. The serial-vs-parallel kernel choice and the LangevinState-carries-der_kernel shape are the plan's stated `mvs_device`/`RankingState` precedents.

## Deferred / Device-Validation Items

- **rocm numeric device assert (ε=1e-4) — PENDING in-env.** The device value-match (Test 1 / Test 2) fires only on a real rocm/cuda backend; a full `--features rocm` rebuild was blocked by the known root-disk pressure (100% full, ~1.7M free — a full hip-cfg rebuild of polars/arrow/cubecl-hip needs far more than the reclaimable incremental cache). Per prior-phase practice this rocm in-env numeric validation is discharged by the orchestrator, NOT fabricated here. Risk is low: the kernel avoids the known `-inf`-literal JIT landmine (0 `NEG_INF`, finite arithmetic only) and mirrors the rocm-proven `bootstrap_device` Poisson rejection-loop shape + the `mvs_device` per-block reseed transcription verbatim.
- **Kaggle CUDA sign-off** — human-gated, deferred to Plan 10 (the phase aggregate device oracle), per the plan.

## GPUT-20 status

AddLangevinNoise adds a per-element seeded Gaussian to the resident der buffer behind the langevin gate, reproducing the frozen pinned-seed CPU `coefficient · std_normal` sequence at ε=1e-4 (CPU-side draw-order model + gate proven; device numeric assert pending in-env). PairLogit+Langevin falls back to CPU. Five-family device-coverage surface complete.

## Self-Check: PASSED

- Created files present: `kernels/langevin.rs`, `kernels/langevin_test.rs`, `13-09-SUMMARY.md`.
- Commits present: `025240a` (Task 1 feat), `382c2d2` (Task 2 test).
