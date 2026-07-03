# SPIKE-REDUCTION — Deterministic GPU Reduction (D-03 / D-04)

**Phase:** 10 — GPU Foundations
**Plan:** 10-03 (reduce family + reduction-determinism spike)
**Requirement:** GPUT-16
**Date:** 2026-07-03
**In-env hardware:** AMD gfx1100 / RDNA3 wave32, ROCm 7.1 (real GPU, no CUDA in-env)
**Evidence source:** `crates/cb-backend/src/kernels/reduce.rs` variance harness
(`reduce_finalize_strategies_are_deterministic_and_report_path`), run in-env.

> **Contract (D-04):** the measured-best deterministic strategy IS the reduce that
> ships — no throwaway prototype. The winner feeds Phase 11's ε=1e-4 histogram gate as
> step 0. This document records the candidate set, the in-env err + determinism
> evidence, per-backend viability, and a firm recommendation.

---

## 1. Candidate set prototyped

Three candidates were implemented as **selectable finalize strategies** on the scalar
cross-cube reduce (the path where cross-cube summation ORDER would otherwise inject
run-to-run nondeterminism). All accumulate in **f64 regardless of channel float type**
(mirrors upstream `update_part_props` `M=8` double re-accumulation, RESEARCH §Reduction).

| # | Candidate | Mechanism | Determinism | Kernel |
|---|-----------|-----------|-------------|--------|
| (a) | **Fixed-order tree reduce** | recursive `block_reduce_kernel` with `use_plane=false` — a fixed shared-mem pairing, no atomics, reduces `n → ⌈n/32⌉ → … → 1` | deterministic on **every** backend (fixed order) | `block_reduce_kernel` (recursed) |
| (b) | **Block-then-host-final-sum** (`HostSumFallback`) | per-cube f64 partials + host `cb_core::sum_f64` (frozen host order) | deterministic (host ordered fold) | `block_reduce_kernel` + host |
| (c) | **Fixed-point i64/u64 atomics** | `round(v·2³⁰) → i64 → u64` bits, `Atomic<u64>::fetch_add`; two's-complement wrapping add == exact signed integer add, order-independent | deterministic **and** higher precision | `block_reduce_fixedpoint_kernel` |

Two further candidates from the RESEARCH table were **not** prototyped (documented, not a
silent cut):

- **Kahan compensation** — reduces error but the summation ORDER still matters; it does
  not by itself buy order-independence, so it is a precision add-on to (a)/(b), not a
  determinism strategy. Deferred (not needed once (a) or (c) gives exact determinism).
- **Sorted-index accumulation** — deterministic only if the sort is stable; it depends on
  the from-scratch radix sort (Plan 10-02) and adds a full sort per reduce. Strictly more
  expensive than (a)/(c) for no determinism gain. Deferred.

---

## 2. In-env err + determinism table (gfx1100, ROCm 7.1)

Input: 300 elements (`≈10` cubes at CUBE_DIM 32) — mixed signs/magnitudes so several
cubes race to finalize. Baseline: `cb_core::sum_f64` = `5595.25`. Each strategy launched
**32×**; `run_to_run_spread = max(result) − min(result)` over the 32 byte-patterns.

| Candidate | Reported path | Runs | Run-to-run spread | abs_div vs baseline | ms (wall) | gfx1100 viable |
|-----------|---------------|------|-------------------|---------------------|-----------|----------------|
| (a) Fixed-order tree reduce | `FixedOrderTree` | 32 | **0** (byte-identical) | `0.000e0` (exact) | see §4 (TBD Kaggle) | ✅ yes |
| (b) Block-then-host-sum | `HostSum` | 32 | **0** (byte-identical) | `0.000e0` (exact) | see §4 (TBD Kaggle) | ✅ yes |
| (c) Fixed-point u64 atomics | `FixedPointAtomic` | 32 | **0** (byte-identical) | `0.000e0` (exact) | see §4 (TBD Kaggle) | ✅ yes (see §3) |

All three candidates showed **zero run-to-run spread** over 32 launches and landed on the
baseline **exactly** (`abs_div = 0.000e0`) for this input. The harness additionally
asserts the **reported path matches the device's advertised capability** — a silent
atomic→deterministic switch fails the test rather than passing (T-10-07).

**Cross-check (existing in-tree oracle):** the f64-atomic finalize
(`launch_block_reduce_atomic_f64`) reports `HostSumFallback` on gfx1100 (no advertised
`Atomic<f64>` add) with spread 0 — consistent with Phase 7.6.

---

## 3. Per-backend viability

| Backend | `Atomic<f64>` add | `Atomic<u64>` add | (a) tree | (b) host-sum | (c) fixed-point |
|---------|-------------------|-------------------|----------|--------------|-----------------|
| **gfx1100 / RDNA3 (in-env)** | **not advertised** (Phase 7.6) | **advertised** ✅ (measured this plan) | ✅ | ✅ | ✅ runs the integer-atomic kernel **in-env** |
| CUDA (Kaggle, authoritative) | advertised (typical) | advertised (typical) | ✅ | ✅ | ✅ — err+ms via 10-09 |

**Key measured finding (updates the plan's assumption):** gfx1100 **advertises
`Atomic<u64>` add** even though it does NOT advertise `Atomic<f64>` add. The plan
anticipated the fixed-point path would only exercise on CUDA (with gfx1100 falling back to
`HostSum`). In practice the **fixed-point u64 kernel runs on-device in-env on gfx1100**
(`f64::round` + `Atomic<u64>::fetch_add` JIT cleanly under HIP/ROCm 7.1) and is
byte-exact. This is exactly the manual's rationale (`09_fixedpoint_atomics.md §5`): "some
backends do not expose `Atomic<f64>` even when they DO support `Atomic<u64>` — which is
why the wide-integer route is preferred." The capability gate remains: on any backend that
does **not** advertise `Atomic<u64>` add, strategy (c) reports the deterministic `HostSum`
downgrade (never a silent switch).

---

## 4. CUDA authoritative numbers — AWAITING KAGGLE (10-09)

The in-env harness measures **correctness + determinism** (err + run-to-run spread), not
isolated per-strategy wall-clock. The authoritative **err + ms** per candidate on CUDA are
filled by the human-gated Kaggle run (Plan 10-09, `bench/cuda_oracle.ipynb`):

| Candidate | CUDA err (≤1e-4) | CUDA ms | Status |
|-----------|------------------|---------|--------|
| (a) Fixed-order tree reduce | **TBD** | **TBD** | awaiting Kaggle CUDA run (10-09) |
| (b) Block-then-host-sum | **TBD** | **TBD** | awaiting Kaggle CUDA run (10-09) |
| (c) Fixed-point u64 atomics | **TBD** | **TBD** | awaiting Kaggle CUDA run (10-09) |

**These numbers are NOT fabricated.** They are populated from the Kaggle notebook that
warms one untimed launch, runs the correctness oracle (blocking), then times each strategy
draining the CubeCL lazy queue before stopping the clock (RESEARCH warm-run caveat).

**Fill source (wired in 10-09):** `bench/cuda_oracle.ipynb` runs the reduce oracle under
`cargo test … reduce -- --nocapture` on CUDA; its
`reduce_finalize_strategies_are_deterministic_and_report_path` output reports per-strategy
run-to-run spread + path + `abs_div`. A dedicated notebook markdown cell ("Fill
SPIKE-REDUCTION.md §4") instructs the human to transcribe the CUDA `err`/`ms` for (a)/(b)/(c)
into the table above; the human sign-off is logged in `bench/RESULTS.md`.

---

## 5. Recommendation (D-04 — the winner that ships)

The reduce family has **two distinct finalize shapes**, and the recommendation differs by
shape:

### 5a. Segmented-reduce / reduce-by-key (one cube per segment) → **Fixed-order f64 tree reduce**

Each segment is summed by a single cube, so there is **no cross-cube contention** — a
fixed-order f64 shared-mem tree reduce is already deterministic and needs no atomics. This
is what `segmented_reduce_kernel` and `reduce_by_key_kernel` **ship with** (this plan).
Recommendation: **keep it** — it is the simplest deterministic option and carries no
backend-capability dependency.

### 5b. Scalar / histogram accumulator (many cubes → one cell) → **Fixed-point u64 atomics**, tree-reduce fallback

For the many-cubes-contend-on-one-cell case that Phase 11's histogram kernel needs, the
recommended winner is **(c) fixed-point i64/u64 atomics**:

- **Single-pass, device-resident** — one kernel launch, no host round-trip (unlike (b))
  and no `log(n)` recursive launches (unlike (a)); keeps the accumulator on device across
  the grow loop.
- **Deterministic AND higher precision** — integer add is exact and order-independent;
  no mantissa erosion (manual §5).
- **Viable on both in-env backends** — gfx1100 advertises `Atomic<u64>` add (measured),
  and CUDA does too. Where a backend lacks it, the harness **downgrades to (a) fixed-order
  tree reduce** (the portable, no-atomic deterministic fallback) — reported explicitly.

**Winner feeding Phase 11:** fixed-point `Atomic<u64>` accumulation (LDS privatization +
fixed-point per manual §08/§09) for the histogram accumulator, with the fixed-order tree
reduce as the capability-fallback. Scale `k = 30` (`2³⁰`) per manual §2.1/§4 (unit-scale
gradient/hessian sums stay well under `2⁶³`). The per-segment reduces ship with the
fixed-order f64 tree reduce.

---

## 6. Traceability

- **Kernels:** `crates/cb-backend/src/kernels.rs` — `segmented_reduce_kernel`,
  `reduce_by_key_kernel` (+ `key_head_flag_kernel`, `segment_offset_scatter_kernel`),
  `block_reduce_fixedpoint_kernel`, `REDUCE_FIXEDPOINT_SCALE_F64`.
- **Harness / oracle:** `crates/cb-backend/src/kernels/reduce.rs` —
  `reduce_finalize_strategies_are_deterministic_and_report_path` (variance + path),
  `segmented_reduce_matches_serial`, `reduce_by_key_matches_serial`.
- **In-env command:** `cargo test -p cb-backend --no-default-features --features rocm reduce`
  → 8/8 green on gfx1100.
- **Threats mitigated:** T-10-06 (nondeterministic result — zero spread over 32 launches),
  T-10-07 (silent capability downgrade — path assertion), T-10-08 (portability UB — no
  `-inf` literal; rocm smoke green).
- **Manual refs:** `09_fixedpoint_atomics.md`, `08_atomic_contention.md`,
  `Cubecl_shared_memory.md`.
