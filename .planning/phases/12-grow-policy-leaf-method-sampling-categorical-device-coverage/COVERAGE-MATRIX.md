# Phase 12 — GPU Device Coverage Matrix (SC-5)

> Per-family GPU device-coverage record for Phase 12 (grow-policy / leaf-method /
> sampling / categorical device coverage). Delivered by Plan 12-09.
>
> **Authority policy (STATE.md / 12-VALIDATION.md):** Kaggle CUDA `--features cuda`
> is the **sole** correctness + speed authority for milestone v1.1. There is **no
> NVIDIA GPU in-env** — the in-env device evidence below is **ROCm gfx1100 self-oracle
> (convenience gate only)**, NOT the correctness authority. Each family's authoritative
> ε=1e-4 correctness sign-off (vs the Rust CPU path) and its BENCH-02 speed measurement
> are **human-gated** Kaggle CUDA notebook runs reusing the Phase-10 `bench/` harness.
>
> **Anti-fabrication (T-12-14 / Phase 11-05 PAUSED precedent):** No Kaggle CUDA
> correctness or speed number appears in this matrix until it is human-reported.
> Every family below is currently **PENDING-KAGGLE**. Per the milestone gate a family
> without a recorded Kaggle sign-off is treated as **`Ok(None)` → CPU fallback**
> regardless of the in-code device arm — it is **not** authoritatively device-covered.

---

## Status legend

- **CPU / in-env self-oracle** — the device path reproduced its Rust CPU reference
  in-env at ε≤1e-4 (or bit-exact). ROCm gfx1100 unless noted. Convenience gate only.
- **ROCm smoke** — `cargo build/test -p cb-backend --features rocm` executed the path in-env.
- **Kaggle CUDA correctness** — the authoritative ε=1e-4 sign-off vs the Rust CPU path.
  `PENDING-KAGGLE` until human-reported (Task 1).
- **BENCH-02 speed** — device vs host-CPU baseline (and vs official CatBoost GPU where
  comparable), warm-run / JIT-excluded, train-only. `PENDING-KAGGLE` until human-reported
  (Task 2); only recordable **after** that family's correctness gate passes.
- **Authoritative gate state** — `Ok(None) → CPU fallback (PENDING-KAGGLE)` until the
  Kaggle sign-off lands, at which point it flips to `device-covered`.

Rows ordered by the retained roadmap sub-order (grow policies → Exact → bootstrap → MVS → CTR) per **D-02**.

---

## Coverage Matrix

| # | Family | Req ID | Plan | In-env device self-oracle (ε, convenience) | ROCm smoke (gfx1100) | Kaggle CUDA correctness (ε=1e-4 vs Rust CPU) | BENCH-02 speed (device vs host-CPU) | Authoritative gate state |
|---|--------|--------|------|---------------------------------------------|----------------------|-----------------------------------------------|-------------------------------------|--------------------------|
| 1 | Depthwise / Lossguide (non-symmetric grow) | GPUT-18 | 12-03 | **PASS** — structure integer-exact vs `leaf_wise_grower`; leaf-value max abs_div **0.000e0** (bit-exact). `nonsym_grow` 4/4, `device_nonsym_fit` 2/2 | ✅ ran in-env (host-driven `pointwise_hist2`, Atomic\<f64\>; not resident → unaffected by the Atomic\<u64\> regression) | **PENDING-KAGGLE** | **PENDING-KAGGLE** (blocked on correctness) | `Ok(None) → CPU fallback (PENDING-KAGGLE)` |
| 2 | Region (grow-policy PATH grow) | GPUT-18 | 12-04 (over 12-02 CPU Region) | **PASS** — path structure EXACT vs frozen Plan-02 CPU Region; leaf-value max abs_div **0.000e0**; `device_region_fit` max \|Δpred\| **0.000e0**. `region_device` 1/1, fit 1/1 | ✅ ran in-env (host-driven; no Atomic\<u64\> needed) | **PENDING-KAGGLE** | **PENDING-KAGGLE** (blocked on correctness) | `Ok(None) → CPU fallback (PENDING-KAGGLE)` |
| 3 | Exact weighted-quantile leaf (Quantile/MAE/MAPE) | GPUT-19 | 12-05 | **PASS (leaf VALUES)** — `device_exact_leaf_delta` == `exact_leaf_delta` ≤1e-4 for Quantile/MAE/MAPE. `exact_quantile` 6/6, `segmented_sort` 4/4. NOTE: structure der is the RMSE-residual der (MVP); upstream quantile-der split parity deferred to Kaggle | ✅ rocm full suite 146/146 (cpu backend cannot run the multi-kernel radix composition by design — cpu-red/rocm-green) | **PENDING-KAGGLE** | **PENDING-KAGGLE** (blocked on correctness) | `Ok(None) → CPU fallback (PENDING-KAGGLE)` |
| 4 | Bootstrap draw (Bernoulli / Bayesian / Poisson) + random-strength | GPUT-09 | 12-06 | **PASS (DRAW)** — Bernoulli **bit-for-bit** vs frozen `TFastRng64`; Bayesian ≤1e-4 (FastLogf approx); Poisson determinism only (no CPU oracle, D-11). `bootstrap` 7/7 | ✅ 7/7 rocm (serial u64 arithmetic, no atomics). e2e resident grow SKIPS on the in-env Atomic\<u64\> regression (WR-01) | **PENDING-KAGGLE** | **PENDING-KAGGLE** (blocked on correctness) | `Ok(None) → CPU fallback (PENDING-KAGGLE)` |
| 5 | Minimal-Variance Sampling (MVS) | GPUT-17 | 12-07 | **PASS** — device weights vs frozen CPU `mvs_sample_weights` max_div **4.4e-16 … 5.3e-15**, kept-counts bit-exact. `mvs` 3/3. NOTE: covers caller-pinned `mvs_lambda`; unpinned-λ declines to CPU | ✅ 3/3 rocm (serial kernel EXECUTES in-env, no atomics). e2e resident grow blocked by the Atomic\<u64\> regression | **PENDING-KAGGLE** | **PENDING-KAGGLE** (blocked on correctness) | `Ok(None) → CPU fallback (PENDING-KAGGLE)` |
| 6 | CTR / permutation-dependent categorical (ordered / one-hot / tensor) | GPUT-10 | 12-08 | **PASS** — device CTR vs CPU `online_ctr_prefix_binclf` good/total **EXACT**, value ≤1e-4; CTR→cindex bit-exact. `ctr` 8/8 on **both cpu and rocm**. NOTE: single-permutation (`fold_count==1`) only; multi-fold → `Ok(None)` (Open Q3) | ✅ 8/8 rocm **and** 8/8 cpu (exact integer prefix counting, no Atomic\<u64\>) | **PENDING-KAGGLE** | **PENDING-KAGGLE** (blocked on correctness) | `Ok(None) → CPU fallback (PENDING-KAGGLE)` |

---

## Recorded Kaggle CUDA tables (verbatim — filled from human-reported checkpoint results)

### Task 1 — Kaggle CUDA ε=1e-4 correctness sign-off (per family)

> Blocking gate. Reused from the Phase-10 Kaggle CUDA oracle harness. Max abs/rel
> divergence of the device path vs the Rust CPU path on the CUDA backend, per frozen
> Phase-12 fixture. **Not yet run — no NVIDIA GPU in-env.**

| Family | Req ID | Frozen fixture | Max divergence (device vs Rust CPU) | Result |
|--------|--------|----------------|-------------------------------------|--------|
| Depthwise / Lossguide | GPUT-18 | Plan-03 non-sym fixture | _pending_ | **PENDING-KAGGLE** |
| Region | GPUT-18 | Plan-02 frozen CPU Region path | _pending_ | **PENDING-KAGGLE** |
| Exact (Quantile/MAE/MAPE) | GPUT-19 | Plan-05 exact-leaf fixture | _pending_ | **PENDING-KAGGLE** |
| Bootstrap | GPUT-09 | Plan-06 frozen `TFastRng64` sample | _pending_ | **PENDING-KAGGLE** |
| MVS | GPUT-17 | Plan-07 frozen MVS sample | _pending_ | **PENDING-KAGGLE** |
| CTR | GPUT-10 | Plan-08 categorical-heavy fixture | _pending_ | **PENDING-KAGGLE** |

### Task 2 — BENCH-02 Kaggle CUDA speed measurement (per family)

> Only recordable **after** the family's Task-1 correctness gate passes. Train-only,
> warm-run, JIT-excluded, lazy CubeCL queue drained with a read-back before stopping
> the clock. **Not yet run — no NVIDIA GPU in-env.**

| Family | Req ID | Device train time | Host-CPU baseline | Official CatBoost GPU (where comparable) | Result |
|--------|--------|-------------------|-------------------|------------------------------------------|--------|
| Depthwise / Lossguide | GPUT-18 | _pending_ | _pending_ | _pending_ | **PENDING-KAGGLE** |
| Region | GPUT-18 | _pending_ | _pending_ | _pending_ | **PENDING-KAGGLE** |
| Exact (Quantile/MAE/MAPE) | GPUT-19 | _pending_ | _pending_ | _pending_ | **PENDING-KAGGLE** |
| Bootstrap | GPUT-09 | _pending_ | _pending_ | _pending_ | **PENDING-KAGGLE** |
| MVS | GPUT-17 | _pending_ | _pending_ | _pending_ | **PENDING-KAGGLE** |
| CTR | GPUT-10 | _pending_ | _pending_ | _pending_ | **PENDING-KAGGLE** |

---

## Phase 12 Success Criteria coverage (ROADMAP)

| SC | Description | Status |
|----|-------------|--------|
| SC-1..3 | Per-family device kernels + CPU Region path + gate arms landed (Plans 01–08) | ✅ landed in-env (self-oracled) |
| SC-4 | Per-family BENCH-02 speed as it lands | ⏳ **PENDING-KAGGLE** (Task 2) |
| SC-5 | Documented per-family GPU coverage matrix (this file) | ✅ scaffolded; correctness/speed cells **PENDING-KAGGLE** |

All six families' authoritative correctness + speed sign-offs (GPUT-18/19/09/17/10 + BENCH-02)
remain **Pending** until the Kaggle CUDA notebook is run and the tables above are filled.

---

## Footer notes

- **D-04 / GPUT-14 no-regression:** the CPU / host training path is **byte-unchanged**.
  Every device family is behind an all-or-nothing coverage gate; the `No`-bootstrap /
  non-covered / unpinned configurations return `Ok(None)` and fall back to the unchanged
  CPU path. Device coverage never mutates the host numerics.
- **Multi-fold / multi-permutation CTR deferral (Open Q3):** only the single-permutation
  CTR regime (`fold_count == 1`) is covered on device. A multi-fold / multi-permutation
  CTR is declined behind `Ok(None)` → CPU fallback and deferred to a later wave — **not**
  fabricated as covered.
- **In-env ROCm `Atomic<u64>`-advertisement regression (deferred-items.md, Plan 12-06):**
  the in-env gfx1100 runtime currently does **not** advertise `Atomic<u64>` add, so the
  resident-histogram depth≥1 **grow** oracles (`session_depth_gt1_grows_and_matches_direct`,
  `session_residency_matches_cpu_multi_tree_boosting`,
  `session_exact_leaf_grows_finite_quantile_leaves`) are red in-env. This is an
  environment/driver capability-state regression (memory `phase10-03` records gfx1100 DID
  advertise it previously), **not** a code regression. It blocks the in-env **e2e resident-grow**
  validation for Exact / bootstrap / MVS (their per-draw / per-leaf **self-oracles** all pass
  in-env using serial or host-driven kernels with no u64 atomics). Depthwise/Lossguide and
  Region are host-driven (Atomic\<f64\> `pointwise_hist2`, not resident) and ran fully in-env.
  This further motivates the Kaggle CUDA authority: the authoritative full-boosting-loop
  correctness + speed sign-off runs on CUDA where the resident histogram path is exercised.
- **Anti-fabrication:** per T-12-14 and the Phase 11-05 PAUSED precedent, no correctness or
  speed number is entered above without a human-reported Kaggle CUDA measurement. This matrix
  is honest as of the pause: **all six families PENDING-KAGGLE**.

---

_Scaffolded by Plan 12-09 (Task 3) autonomously. Tasks 1 & 2 (Kaggle CUDA correctness +
speed sign-off) are human-gated — see `12-09-PLAN.md`. A continuation agent fills the two
tables above and flips each signed-off family from `Ok(None) → CPU fallback (PENDING-KAGGLE)`
to `device-covered` once the human pastes the recorded results._
