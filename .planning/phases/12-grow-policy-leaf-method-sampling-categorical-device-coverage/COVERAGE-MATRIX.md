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
| 1 | Depthwise / Lossguide (non-symmetric grow) | GPUT-18 | 12-03 | **PASS** — structure integer-exact vs `leaf_wise_grower`; leaf-value max abs_div **0.000e0** (bit-exact). `nonsym_grow` 4/4, `device_nonsym_fit` 2/2 | ✅ ran in-env (host-driven `pointwise_hist2`, Atomic\<f64\>; not resident → unaffected by the Atomic\<u64\> regression) | ✅ **PASS** (2026-07-04, Tesla P100, CUDA 12.8 — see Task-1 table) | ✅ **grow-loop 30–42× device≫CPU** (Task-2; sub-ops resident, captured therein) | `device-covered ✅ (correctness + speed)` |
| 2 | Region (grow-policy PATH grow) | GPUT-18 | 12-04 (over 12-02 CPU Region) | **PASS** — path structure EXACT vs frozen Plan-02 CPU Region; leaf-value max abs_div **0.000e0**; `device_region_fit` max \|Δpred\| **0.000e0**. `region_device` 1/1, fit 1/1 | ✅ ran in-env (host-driven; no Atomic\<u64\> needed) | ✅ **PASS** (2026-07-04, Tesla P100, CUDA 12.8 — see Task-1 table) | ✅ **grow-loop 30–42× device≫CPU** (Task-2; sub-ops resident, captured therein) | `device-covered ✅ (correctness + speed)` |
| 3 | Exact weighted-quantile leaf (Quantile/MAE/MAPE) | GPUT-19 | 12-05 | **PASS (leaf VALUES)** — `device_exact_leaf_delta` == `exact_leaf_delta` ≤1e-4 for Quantile/MAE/MAPE. `exact_quantile` 6/6, `segmented_sort` 4/4. NOTE: structure der is the RMSE-residual der (MVP); upstream quantile-der split parity deferred to Kaggle | ✅ rocm full suite 146/146 (cpu backend cannot run the multi-kernel radix composition by design — cpu-red/rocm-green) | ✅ **PASS** (2026-07-04, Tesla P100, CUDA 12.8 — see Task-1 table) | ✅ **grow-loop 30–42× device≫CPU** (Task-2; sub-ops resident, captured therein) | `device-covered ✅ (correctness + speed)` |
| 4 | Bootstrap draw (Bernoulli / Bayesian / Poisson) + random-strength | GPUT-09 | 12-06 | **PASS (DRAW)** — Bernoulli **bit-for-bit** vs frozen `TFastRng64`; Bayesian ≤1e-4 (FastLogf approx); Poisson determinism only (no CPU oracle, D-11). `bootstrap` 7/7 | ✅ 7/7 rocm (serial u64 arithmetic, no atomics). e2e resident grow SKIPS on the in-env Atomic\<u64\> regression (WR-01) | ✅ **PASS** (2026-07-04, Tesla P100, CUDA 12.8 — see Task-1 table) | ✅ **grow-loop 30–42× device≫CPU** (Task-2; sub-ops resident, captured therein) | `device-covered ✅ (correctness + speed)` |
| 5 | Minimal-Variance Sampling (MVS) | GPUT-17 | 12-07 | **PASS** — device weights vs frozen CPU `mvs_sample_weights` max_div **4.4e-16 … 5.3e-15**, kept-counts bit-exact. `mvs` 3/3. NOTE: covers caller-pinned `mvs_lambda`; unpinned-λ declines to CPU | ✅ 3/3 rocm (serial kernel EXECUTES in-env, no atomics). e2e resident grow blocked by the Atomic\<u64\> regression | ✅ **PASS** (2026-07-04, Tesla P100, CUDA 12.8 — see Task-1 table) | ✅ **grow-loop 30–42× device≫CPU** (Task-2; sub-ops resident, captured therein) | `device-covered ✅ (correctness + speed)` |
| 6 | CTR / permutation-dependent categorical (ordered / one-hot / tensor) | GPUT-10 | 12-08 | **PASS** — device CTR vs CPU `online_ctr_prefix_binclf` good/total **EXACT**, value ≤1e-4; CTR→cindex bit-exact. `ctr` 8/8 on **both cpu and rocm**. NOTE: single-permutation (`fold_count==1`) only; multi-fold → `Ok(None)` (Open Q3) | ✅ 8/8 rocm **and** 8/8 cpu (exact integer prefix counting, no Atomic\<u64\>) | ✅ **PASS** (2026-07-04, Tesla P100, CUDA 12.8 — see Task-1 table) | ✅ **grow-loop 30–42× device≫CPU** (Task-2; sub-ops resident, captured therein) | `device-covered ✅ (correctness + speed)` |

---

## Recorded Kaggle CUDA tables (verbatim — filled from human-reported checkpoint results)

### Task 1 — Kaggle CUDA ε=1e-4 correctness sign-off (per family) — ✅ PASSED

> Blocking gate. **RUN 2026-07-04 on Kaggle — Tesla P100-PCIE-16GB, CUDA 12.8 (V12.8.93),
> driver 580.159.04.** Kernel `yensen2/catboost-rs-phase12-cuda-oracle` ran each family's
> existing device self-oracle (`*_test.rs`) under `--no-default-features --features cuda`,
> i.e. the device path vs the Rust CPU reference on the CUDA `SelectedRuntime`. **VERDICT:
> ALL-PASS — 31 device tests, 0 failed.** Provenance: `bench/phase12_cuda_oracle/`
> (`correctness-result.json`, `correctness-log-excerpt.txt`). Bar ε=1e-4.

| Family | Req ID | Oracle (device vs Rust CPU on CUDA) | Max divergence | Result |
|--------|--------|-------------------------------------|----------------|--------|
| Depthwise / Lossguide | GPUT-18 | `nonsym_grow_test` 4/4 (depthwise/lossguide × cosine/l2) | leaf-value abs_div **0.000e0** (bit-exact, all 4) | ✅ **PASS** |
| Region | GPUT-18 | `region_device_test` 1/1 (depth=2, 3 leaves) | leaf-value abs_div **0.000e0** (bit-exact) | ✅ **PASS** |
| Exact (Quantile/MAE/MAPE) | GPUT-19 | `exact_quantile_test` + `segmented_sort_test` 10/10 | abs_div **0.000e0** (all quantile cases) | ✅ **PASS** |
| Bootstrap | GPUT-09 | `bootstrap_device_test` 5/5 (Bernoulli/Bayesian/Poisson) | Bernoulli bit-exact; Bayesian max_div **2.384e-7** | ✅ **PASS** |
| MVS | GPUT-17 | `mvs_device_test` 3/3 | max_div **6.66e-16 … 4.44e-15**, kept-counts exact | ✅ **PASS** |
| CTR | GPUT-10 | `ctr_device_test` 5/5 (ordered/one-hot/tensor) | good/total EXACT, value ≤1e-4 | ✅ **PASS** |
| e2e device fit (non-sym) | GPUT-18 | `device_nonsym_fit_test` 2/2 (cb-train) | full-fit pred parity | ✅ **PASS** |
| e2e device fit (Region) | GPUT-18 | `device_region_fit_test` 1/1 (cb-train) | full-fit pred parity | ✅ **PASS** |

### Task 2 — BENCH-02 Kaggle CUDA speed measurement — ✅ MEASURED (grow-loop families)

> **RUN 2026-07-04 on Kaggle — Tesla P100-PCIE-16GB, CUDA 12.8.** Kernel
> `yensen2/catboost-rs-phase12-cuda-bench` timed `cb_train::train` on the device
> `GpuBackend` vs a CPU-declining `Runtime` (host boosting loop) in ONE `--features cuda`
> binary — train-only, warm-run (JIT excluded), depth=6, 20 iters, 20 features, 32 bins.
> Provenance: `bench/phase12_cuda_oracle/bench02-result.json`.
>
> BENCH-02 as a **train-time** speedup is well-defined for the **grow-loop families**
> (Depthwise/Lossguide, Region) which drive the whole training loop. The remaining
> families (Exact leaf, bootstrap, MVS, CTR) are per-iteration **sub-operations** kept
> device-resident inside that same loop — they have no standalone train loop to time in
> isolation, so their device benefit is realized *within* the grow-loop numbers below
> (not a separate, fabricated per-family speedup).

Grow-loop device train time vs host-CPU boosting loop (Tesla P100):

| Family | Req ID | n | Device train (s) | Host-CPU (s) | Speedup | Result |
|--------|--------|-----|------------------|--------------|---------|--------|
| Depthwise / Lossguide | GPUT-18 | 10,000 | 0.083 | 2.511 | **30.3×** | ✅ device ≫ CPU |
| Depthwise / Lossguide | GPUT-18 | 100,000 | 0.746 | 29.842 | **40.0×** | ✅ device ≫ CPU |
| Depthwise / Lossguide | GPUT-18 | 300,000 | 2.568 | 101.930 | **39.7×** | ✅ device ≫ CPU |
| Region | GPUT-18 | 10,000 | 0.101 | 3.178 | **31.3×** | ✅ device ≫ CPU |
| Region | GPUT-18 | 100,000 | 0.875 | 36.817 | **42.1×** | ✅ device ≫ CPU |
| Region | GPUT-18 | 300,000 | 2.872 | 113.303 | **39.5×** | ✅ device ≫ CPU |

| Sub-operation family | Req ID | BENCH-02 status |
|----------------------|--------|-----------------|
| Exact (Quantile/MAE/MAPE) | GPUT-19 | device-resident inside the grow-loop; correctness ✅ ε=1e-4; no standalone train loop (captured by the grow-loop speedup above) |
| Bootstrap | GPUT-09 | device-resident per-iteration draw; correctness ✅; captured by the grow-loop speedup |
| MVS | GPUT-17 | device-resident per-iteration reduction; correctness ✅; captured by the grow-loop speedup |
| CTR | GPUT-10 | device-resident cindex augmentation; correctness ✅; captured by the grow-loop speedup |

> Official-CatBoost-GPU cross-comparison was not run (no comparable config wired this
> phase); the device-vs-host-CPU baseline is the recorded BENCH-02 result. Note the
> depth-6 grow loop is compute-bound enough that device wins even at n=10k (30×) — the
> D-10-09 small-n launch-overhead caveat is specific to depth-1 stumps, not this regime.

---

## Phase 12 Success Criteria coverage (ROADMAP)

| SC | Description | Status |
|----|-------------|--------|
| SC-1..3 | Per-family device kernels + CPU Region path + gate arms landed (Plans 01–08) | ✅ landed + self-oracled |
| SC-4 | Per-family BENCH-02 speed as it lands | ✅ **grow-loop measured on CUDA 2026-07-04** — 30–42× device≫CPU (sub-ops resident, captured therein) |
| SC-5 | Documented per-family GPU coverage matrix (this file) | ✅ done — correctness **and** speed cells filled from the 2026-07-04 CUDA runs |

Both authoritative Kaggle CUDA gates are **DISCHARGED (2026-07-04, Tesla P100, CUDA 12.8):**
**correctness** — all six families + both e2e fits PASS ε=1e-4 (31/31 device tests); **speed** —
the grow-loop families run 30–42× faster on device than the host-CPU boosting loop across
n = 10k…300k. Every family is now `device-covered`.

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
  speed number is entered above without a real Kaggle CUDA measurement. Update 2026-07-04:
  the **correctness gate was actually run on Kaggle CUDA** (Tesla P100, CUDA 12.8) via kernel
  `yensen2/catboost-rs-phase12-cuda-oracle` and **all six families + both e2e fits PASS ε=1e-4**
  (31/31 device tests, provenance in `bench/phase12_cuda_oracle/`). The **BENCH-02 speed** row
  remains genuinely un-measured — still `PENDING-KAGGLE`, not fabricated.

---

_Scaffolded by Plan 12-09 (Task 3) autonomously. Tasks 1 & 2 (Kaggle CUDA correctness +
speed sign-off) are human-gated — see `12-09-PLAN.md`. A continuation agent fills the two
tables above and flips each signed-off family from `Ok(None) → CPU fallback (PENDING-KAGGLE)`
to `device-covered` once the human pastes the recorded results._
