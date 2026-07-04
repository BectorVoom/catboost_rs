# Phase 13 — GPU Device Coverage Matrix (SC-5)

> Per-family GPU device-coverage record for Phase 13 (pairwise / ranking / multiclass /
> ordered / langevin loss-family coverage). Delivered by Plan 13-10.
>
> **Authority policy (STATE.md / 13-VALIDATION.md):** Kaggle CUDA `--features cuda` is the
> **sole** correctness + speed authority for milestone v1.1. There is **no NVIDIA GPU
> in-env** — the in-env device evidence below is **ROCm gfx1100 self-oracle (convenience
> gate only)**, NOT the correctness authority. Each family's authoritative ε=1e-4
> correctness sign-off (vs the Rust CPU path) is a **human-gated** Kaggle CUDA notebook run
> reusing the Phase-10/12 harness (`bench/kaggle_cuda_phase13.ipynb`).
>
> **Anti-fabrication (T-13-19 / Phase 11-05 & 12-09 PAUSED precedent):** No Kaggle CUDA
> correctness or speed number appears in this matrix until it is human-reported from a real
> CUDA run. Every family below is currently **PENDING-KAGGLE**. Correctness is a BLOCKING
> gate before any speed number.

---

## The `Ok(None)` reality this phase (read before interpreting the BENCH-02 column)

All five Phase-13 families land their **device der-driver / solver / grouping / noise kernel
+ a self-oracle + a structural coverage-gate seam** (`PairwiseState` / `RankingState` /
`MulticlassState` / `OrderedState` / `LangevinState`), but the session `begin()` currently
**declines to `Ok(None)`→CPU** for every one of them. The per-tree grow seam each family needs
(pairwise pair/group descriptor, ranking query descriptor, multi-dim block grow, ordered
permutation descriptor, langevin config knob) is a **forward dependency** carried by
`grow_tree_on_device` (which passes only approx/target today). This mirrors the Plan-01
pairwise precedent adopted uniformly across Plans 01–09.

Consequence for BENCH-02: there is **no end-to-end device train loop to time per family this
phase** — exactly Phase-12's *sub-operation family* situation. The device kernels are exercised
+ correctness-gated by the self-oracle; the standalone train-only speedup is realized in the
shared device grow loop (Phase-11 depth-6 grow-loop) and **aggregated in Phase 14**. The
per-family BENCH-02 cell is `captured-by-grow-loop`, **not** a fabricated per-family end-to-end
number. Each family's recordable Phase-13 gate is its **correctness ε=1e-4 device-vs-Rust-CPU
sign-off.**

---

## Status legend

- **In-env device self-oracle (ε, convenience)** — the device kernel reproduced its Rust CPU
  reference in-env at ε≤1e-4 (or bit-exact). ROCm gfx1100 unless noted. Convenience gate only.
- **Kaggle CUDA correctness** — the authoritative ε=1e-4 sign-off vs the Rust CPU path.
  `PENDING-KAGGLE` until human-reported (Task 2).
- **BENCH-02 speed** — per the `Ok(None)` reality above, `captured-by-grow-loop`; the shared
  depth-6 device grow-loop anchor number is `PENDING-KAGGLE` until human-reported.
- **Authoritative gate state** — `Ok(None) → CPU fallback (PENDING-KAGGLE)` until the Kaggle
  correctness sign-off lands, at which point it flips to
  `device-covered (correctness) / grow-seam-pending (speed)`.

Rows ordered by the roadmap sub-order (pairwise+solver → ranking → multiclass → ordered →
langevin) per **D-01**.

---

## Coverage Matrix

| # | Family | Req ID | Plans | In-env device self-oracle (ε, convenience) | Kaggle CUDA correctness (ε=1e-4 vs Rust CPU) | BENCH-02 speed | Authoritative gate state |
|---|--------|--------|-------|---------------------------------------------|-----------------------------------------------|----------------|--------------------------|
| 1 | Pairwise (PairLogit) — per-leaf linear-system assembly + batched f64 Cholesky solver wired into the split scorer | GPUT-11, GPUT-21 | 01, 02 | **PASS** — packed lower-tri `linearSystem` bit-exact vs CPU `calculate_pairwise_leaf_values`; batched f64 Cholesky leaf-values + `CalcScoresCholesky` bit-for-bit vs `cholesky_solve` (non-PD → zeros fallback). `pairwise_deriv` + `cholesky_solve` self-oracled on rocm gfx1100. wgpu (no f64) retains the host scorer | ⏳ **PENDING-KAGGLE** (Task 2) | ⏳ `captured-by-grow-loop` — session `Ok(None)` (grow seam forward dependency) | `Ok(None) → CPU fallback (PENDING-KAGGLE)` |
| 2 | Ranking (query/listwise) — device query-grouping infra + QueryRMSE/QuerySoftMax/QueryCrossEntropy + stochastic YetiRank/PFound-F | GPUT-22 | 03, 04, 05 | **PASS** — group der/weight sums via k=30 fixed-point det path vs CPU `ranking_der` ≤1e-4; QueryRMSE/QuerySoftMax der ≤1e-4; YetiRank/PFound-F **bit-exact (max_div 0.000e0)** vs `yetirank_sample_pairs`+`calc_ders_for_queries`. `query_helper`+`ranking_det`+`ranking_stoch` on rocm gfx1100. **QueryCrossEntropy independently `Ok(None)`** (Open Q3 — no cb_compute der oracle) | ⏳ **PENDING-KAGGLE** (Task 2) | ⏳ `captured-by-grow-loop` — session `Ok(None)` (grow seam forward dependency) | `Ok(None) → CPU fallback (PENDING-KAGGLE)` |
| 3 | Multiclass / multi-output / uncertainty — approx_dim block leaves + K-dim Newton der2 block solve (5 losses) | GPUT-12 | 06, 07 | **PASS** — coupled softmax K=3 + diagonal RMSEWithUncertainty K=2 + diagonal MultiClassOneVsAll K=3 block leaves == CPU `solve_symmetric_newton` ≤1e-4; scalar byte-unchanged at approx_dim==1 (D-04). `multiclass`+`multi_newton` on rocm gfx1100. MultiRMSE classified but has no `Loss` variant yet | ⏳ **PENDING-KAGGLE** (Task 2) | ⏳ `captured-by-grow-loop` — session `Ok(None)` (multi-dim grow seam forward dependency) | `Ok(None) → CPU fallback (PENDING-KAGGLE)` |
| 4 | Ordered boosting (EBoostingType::Ordered) — device-resident per-permutation historical-approx trajectory | GPUT-13 | 08 | **PASS** — resident trajectory fold via `apply_leaf_delta` (identity map + unit rate, one final read-back) **bit-for-bit** vs frozen CPU `ordered_approx_delta_simple` ≤1e-4. `ordered` on rocm gfx1100. Per-object delta host-computed (sequential permutation scan); only the fold runs on device (D-05) | ⏳ **PENDING-KAGGLE** (Task 2) | ⏳ `captured-by-grow-loop` — session `Ok(None)` (permutation-descriptor grow seam forward dependency) | `Ok(None) → CPU fallback (PENDING-KAGGLE)` |
| 5 | Langevin / SGLB — `AddLangevinNoise` per-element seeded Gaussian on the resident reduced der | GPUT-20 | 09 | **PASS** — in-place `der[i] += coefficient · std_normal(seed_i)` (Marsaglia-polar draw order inline) **bit-for-bit** vs frozen pinned-seed CPU sequence ≤1e-4; no `read_one` (D-08). `langevin` on rocm gfx1100. f64/u64 wgpu typed reject (WR-02). `*Pairwise + Langevin` → CPU (A4) | ⏳ **PENDING-KAGGLE** (Task 2) | ⏳ `captured-by-grow-loop` — session `Ok(None)` (grow seam forward dependency; no device Langevin config knob yet) | `Ok(None) → CPU fallback (PENDING-KAGGLE)` |

---

## Recorded Kaggle CUDA tables (verbatim — filled from the human-reported checkpoint results)

### Task 2 — Kaggle CUDA ε=1e-4 correctness sign-off (per family) — ⏳ PENDING-KAGGLE

> **Blocking gate.** RUN on Kaggle — **GPU: _PENDING_ (paste `nvidia-smi` name), CUDA _PENDING_,
> driver _PENDING_.** Kernel `_PENDING_` runs each family's existing device self-oracle
> (`*_test.rs`) under `--no-default-features --features cuda`, i.e. the device path vs the Rust
> CPU reference on the CUDA `SelectedRuntime` (`bench/kaggle_cuda_phase13.ipynb`, correctness
> cell). Bar ε=1e-4. **Do NOT fill until the real run reports.**

| Family | Req ID | Oracle (device vs Rust CPU on CUDA) | Max divergence | Result |
|--------|--------|-------------------------------------|----------------|--------|
| Pairwise | GPUT-11, GPUT-21 | `pairwise_deriv_test` + `cholesky_solve_test` | _PENDING_ | ⏳ PENDING-KAGGLE |
| Ranking | GPUT-22 | `query_helper_test` + `ranking_det_test` + `ranking_stoch_test` | _PENDING_ | ⏳ PENDING-KAGGLE |
| Multiclass | GPUT-12 | `multiclass_test` + `multi_newton_test` | _PENDING_ | ⏳ PENDING-KAGGLE |
| Ordered | GPUT-13 | `ordered_test` | _PENDING_ | ⏳ PENDING-KAGGLE |
| Langevin | GPUT-20 | `langevin_test` | _PENDING_ | ⏳ PENDING-KAGGLE |

### Task 2 — BENCH-02 Kaggle CUDA speed measurement — ⏳ PENDING-KAGGLE

> Only recordable **after** the correctness gate passes (T-13-19). Per the `Ok(None)` reality,
> the per-family end-to-end device train-only speedup is **not independently measurable this
> phase** (session `Ok(None)`; grow seam forward dependency). The recordable anchor is the
> shared depth-6 device grow-loop (`bench/kaggle_cuda_phase13.ipynb`, BENCH-02 cell) — device
> vs host-CPU baseline, warm-run/JIT-excluded, train-only, queue-drained. The host-CPU baseline
> needs a separate cpu-feature wheel run (compile-time features, CLAUDE.md).

| Anchor | Config | Device train (s) | Host-CPU (s) | Speedup | Result |
|--------|--------|------------------|--------------|---------|--------|
| Shared depth-6 device grow-loop (DEPTH6_SPEED_CONFIG, the workload these families feed) | depth=6, 20 iters, 10k×50 | _PENDING_ | _PENDING_ (cpu wheel) | _PENDING_ | ⏳ PENDING-KAGGLE |

| Family | Req ID | BENCH-02 status |
|--------|--------|-----------------|
| Pairwise | GPUT-11, GPUT-21 | device solver/assembly resident; correctness ⏳; standalone train loop pending the per-tree grow seam (captured by the grow-loop anchor / Phase-14 aggregate) |
| Ranking | GPUT-22 | device grouping+der resident; correctness ⏳; standalone train loop pending the query-descriptor grow seam |
| Multiclass | GPUT-12 | device block-leaf Newton resident; correctness ⏳; standalone train loop pending the multi-dim grow seam |
| Ordered | GPUT-13 | device trajectory fold resident; correctness ⏳; standalone train loop pending the permutation-descriptor grow seam |
| Langevin | GPUT-20 | device in-place noise on the resident der; correctness ⏳; standalone train loop pending the grow seam |

---

## Phase 13 Success Criteria coverage (ROADMAP)

| SC | Description | Status |
|----|-------------|--------|
| SC-1..3 | Per-family device kernels + coverage-gate seams landed + self-oracled (Plans 01–09) | ✅ landed + self-oracled in-env (rocm gfx1100) |
| SC-4 | Per-family BENCH-02 speed as it lands | ⏳ **PENDING-KAGGLE** — `captured-by-grow-loop` (session `Ok(None)`; standalone per-family train loop pends the grow seam; anchored on the shared depth-6 grow-loop, aggregated in Phase 14) |
| SC-5 | Documented per-family GPU coverage matrix (this file) | ✅ scaffold done — correctness + speed cells `PENDING-KAGGLE` awaiting the human-gated CUDA run |

Both authoritative Kaggle CUDA gates are **PENDING** for this phase: **correctness** (per-family
ε=1e-4 device-vs-Rust-CPU) and **speed** (BENCH-02 grow-loop anchor) are filled only from a real
`bench/kaggle_cuda_phase13.ipynb` run. Do **not** mark Phase 13 device coverage authoritatively
complete until this file's tables are filled from measured CUDA results.

---

## Footer notes

- **D-04 / GPUT-14 no-regression:** the CPU / host training path is **byte-unchanged**. Every
  device family is behind an all-or-nothing coverage gate; because every family declines to
  `Ok(None)` this phase, real fits transparently run the unchanged CPU path. Device coverage
  never mutates the host numerics. Full CPU suite (`cargo test -p cb-train -p cb-compute`) is the
  no-regression check.
- **Uniform per-family `Ok(None)` (grow seam forward dependency):** unlike Phase 12 (where
  Depthwise/Region flipped to `device-covered` end-to-end), **no** Phase-13 family constructs a
  covered device session yet — the per-tree grow seam that would carry each family's descriptor
  is a forward dependency. The der-driver / solver / self-oracle + structural coverage seam are
  this phase's deliverable; the session flip to `Ok(Some)` lands when the grow seam is wired.
- **QueryCrossEntropy independent deferral (ranking Open Q3):** QueryCrossEntropy has no
  `cb_compute::ranking_der` CPU der oracle, so its bounded shift-search der is landed structurally
  but gated OFF (`ranking_objective_covered == false`) — **not** fabricated as covered.
  QueryRMSE/QuerySoftMax + YetiRank/PFound-F ship regardless.
- **MultiRMSE:** classified for the multiclass diagonal arm but has no `Loss` variant yet; the
  self-oracle uses MultiClass/RMSEWithUncertainty/MultiClassOneVsAll (both hessian structures).
- **In-env ROCm `Atomic<u64>` regression (Phase-12 deferred-items precedent):** where a family's
  self-oracle uses the resident fixed-point reduction, the in-env gfx1100 runtime may not
  advertise `Atomic<u64>` add; those specific resident paths are red in-env by environment/driver
  state, not code regression. This further motivates the Kaggle CUDA authority.
- **Anti-fabrication:** per T-13-19 and the Phase 11-05 / 12-09 PAUSED precedent, no correctness
  or speed number is entered above without a real Kaggle CUDA measurement. Every table cell is
  `PENDING-KAGGLE` / `_PENDING_` until the human pastes the recorded result.

---

_Scaffolded by Plan 13-10 (Task 1) autonomously. Task 2 (Kaggle CUDA correctness + BENCH-02
sign-off) is human-gated — see `13-10-PLAN.md`. A continuation agent fills the two tables above
and flips each signed-off family from `Ok(None) → CPU fallback (PENDING-KAGGLE)` to
`device-covered (correctness) / grow-seam-pending (speed)` once the human pastes the recorded
results._
