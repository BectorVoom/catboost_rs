# Phase 15 — RV-13 Per-Hazard Evidence & Single-Session CUDA Oracle of Record

**Assembled:** 2026-07-05 (Wave 3, plan 15-04)
**Purpose (D-10/HARD-03):** one honest entry per latent parity hazard RV-13-01..04
recording *what diverged*, *the demonstrating oracle*, *the fix*, and *the passing result*.
**Purpose (HARD-01/HARD-02):** record the single authoritative Tesla P100 CUDA session
that gates every v1.1 device family at ε=1e-4 and emits the 12 BENCH-02 depth rows.

Every result below traces to the committed artifacts:
`.planning/phases/15-.../15-01-SUMMARY.md`, `15-02-SUMMARY.md`, and the single-session
record `bench/phase15_cuda_oracle/result.json` (`correctness_verdict: ALL-PASS`,
`single_session: true`, committed at `734109a`).

---

## Per-hazard evidence (RV-13-01..04 — HARD-03)

### RV-13-01 — QuerySoftMax / YetiRank tie-order stability (CONFIRMATORY, A1)

- **What diverged:** *Nothing — this is a confirmatory discharge.* The review flagged
  `descending_order_per_query` (ranking.rs) as a *potential* tie-order parity hazard: if the
  device stable radix did not preserve original-index order on exact ties, the per-query
  descending sort could diverge from the CPU `sort_by(value desc, index asc)`. The oracle
  was written FIRST (Pitfall 1 / A1) to settle whether a divergence actually existed.
- **The oracle:** `tie_order_matches_cpu_stable_descending` in
  `crates/cb-backend/src/gpu_runtime/ranking_stoch_test.rs`. A 2-query fixture with
  deliberate exact ties (including a 3-way tie) is fed to `descending_order_per_query`; the
  returned permutation is asserted equal to an INDEPENDENT CPU stable-descending reference
  (`cpu_stable_descending_order`, `sort_by (value desc, index asc)`), expected
  `[3,0,1,2,4,7,5,6,8]`.
- **The fix:** *verified stable + oracle added* (no body change). The existing
  complemented-key stable radix (`segmented_radix_sort`, carrying `idx0` through both passes)
  already preserves tie order; only a `pub(crate)` visibility bump + a doc-anchored
  tie-stability contract note were added. This is a valid HARD-03 discharge per A1/D-01 — no
  speculative churn to working code.
- **The passing result:** exact permutation equality on the real gfx1100 device (in-env
  rocm smoke) AND in the single-session P100 CUDA run — `tie_order_matches_cpu_stable_descending`
  is listed in `result.json` `rv13_oracles_seen` (ranking family, GPUT-22). The exact-order
  assert is device-gated (`descending_order_per_query` → `plane_inclusive_sum`, unsupported
  on the cubecl cpu backend; record-only on cpu, WR-01). Commit `e77561d` (15-01 Task 1).

### RV-13-02 — QuerySoftMax exp-shift seeded weight-blind (REAL FIX)

- **What diverged:** `query_softmax_ders_host` (ranking.rs) seeded the per-query exp-shift
  max via `compute_group_max_host(approx, …)` — a **weight-BLIND** max over ALL objects,
  whereas CPU `ranking_der.rs:257-266` takes the max over **weight>0 objects only**. On a
  query whose global-max-approx doc has weight ≤ 0, the seed (hence the exp-shift) differed
  from CPU. (Numerically the QuerySoftMax der is shift-invariant — a common `exp(-β·Δmax)`
  factor cancels in the `p = exp/Σexp` share — so the *der* matches to ~1e-16 regardless;
  the divergence is a semantic/exact-seed-parity gap, not a catastrophic value error.)
- **The oracle:** `softmax_weight_max_seed` in `ranking_stoch_test.rs`. Because the der is
  shift-invariant, the oracle's TEETH are a **direct seed-selection assert**:
  `compute_group_max_weighted_host` on a weighted query whose max-approx doc has weight 0
  must select the weight>0 max (`[1.0, 0.8]`), where the pre-fix weight-blind seed would
  have been `3.0`. Backed by a device-gated der1/der2 parity assert vs the frozen CPU der.
- **The fix:** replaced the weight-blind seed with a new `pub(crate)
  compute_group_max_weighted_host` that maxes over `w>0` objects only (seeding `f64::MIN`
  for an all-w≤0 query, relying on the intact downstream `sum_weighted_targets > 0`
  short-circuit), mirroring CPU exactly. Host-side, no `#[cube]` kernel edit (Open Q2).
- **The passing result:** oracle green on cpu (record-only) and on the real rocm gfx1100
  device (der1 max_div 1.1e-16, der2 0.0); listed in `result.json` `rv13_oracles_seen`
  (ranking family, GPUT-22) in the single-session P100 run. Commit `e71d6dd` (15-01 Task 2).
  *Latent (out of scope, noted):* a pathological-magnitude fixture (gap ≳ 745) that would
  make the pre-fix seed underflow also makes the shared `#[cube]` softmax kernel overflow
  `inf·0 = NaN` (the sum_exp loop does not skip w≤0) — a future kernel-hardening item, not a
  HARD-03 blocker.

### RV-13-03 — `compute_group_means_host` n==0 empty-group residency fault (REAL FIX)

- **What diverged:** `compute_group_means_host` (query_helper.rs) guarded `n_groups==0` but
  NOT `n==0`. An all-empty-group offset (`q_offsets=[0,0]`, `n==0`, `n_groups==1`) would fall
  through to `selected_client()` / `client.create` and bind a **zero-length device buffer** —
  the project HIP residency lesson (a device fault, invisible on the cpu "device").
- **The oracle:** `empty_group_means_no_fault` in
  `crates/cb-backend/src/kernels/query_helper_test.rs`. Asserts
  `compute_group_means_host(&[], &[], &[0,0]) == Ok(vec![0.0])` — the RIGHT length (1) AND
  value (0.0), not merely "no panic" (Pitfall 3), and that no zero-length device handle is
  launched.
- **The fix:** an `n == 0` short-circuit returning `vec![0.0; n_groups]` inserted BEFORE any
  `selected_client()` / `client.create` (Pattern B — host-side residency guard), mirroring
  the sibling `remove_group_means_host` guard.
- **The passing result:** authoritatively validated on rocm gfx1100 in-env (D-03):
  `device_backend_active=true`, `empty-group means = [0.0]`, PASS — no zero-length-handle
  fault. Also PASS on cpu. Listed in `result.json` `rv13_oracles_seen` (ranking family) in
  the single-session P100 run. Commit `b7e2e52` (15-02 Task 1).

### RV-13-04 — Pairwise Cholesky near-equal-border tie-break flip (REAL FIX)

- **What diverged:** `select_best_split_over_scores` (pairwise.rs) chose the best border with
  an exact-`==` f64 tie-break. The device-Cholesky path (`not(wgpu)`, f64) and the frozen
  wgpu host-scorer path (`calculate_pairwise_score`, f32/host accumulation) land ~1e-13 apart
  on genuinely-tied borders, so the exact-`==` rule could select DIFFERENT borders across the
  two accumulation orders (a non-deterministic split flip).
- **The oracle:** `pairwise_near_equal_border_tiebreak` in
  `crates/cb-backend/src/kernels/cholesky_solve_test.rs`. Constructs two borders whose true
  scores fall inside the tolerance band; asserts the device-Cholesky path and the frozen
  host-scorer path select the SAME border. Non-tautological: the test ALSO runs the retired
  exact-`==` `argmax` and proves it FLIPS (device→0, host→1) while the new rule agrees.
- **The fix:** replaced the exact-`==` argmax with `pub(crate) select_best_candidate`, a
  near-equal-tolerant (`|a-b| <= REL_TOL·max(|a|,|b|,1)`, `REL_TOL = 1e-9`),
  lowest-index-deterministic selector shared by production and oracle (A2/D-02 — the fix is
  the tie-break, NOT forcing the two solves bit-identical).
- **The passing result:** rocm gfx1100 in-env smoke — `device=0 host=0 (exact flip 0->1);
  separated winner=1 (REL_TOL=1e-9)`, PASS; also PASS on cpu. Listed in `result.json`
  `rv13_oracles_seen` (pairwise family, GPUT-11/21) in the single-session P100 run. Commit
  `cdb3022` (15-02 Task 2).

**HARD-03 status:** all four hazards discharged with evidence (RV-13-01 confirmatory;
RV-13-02/03/04 real fixes), and all four oracles were counted in the single-session CUDA run
(`rv13_oracles_expected == rv13_oracles_seen`, 4/4).

---

## Part A — Single-session correctness verdict (HARD-01)

One `--features cuda` session on Tesla P100 ran all 13 v1.1 device families;
`correctness_verdict: ALL-PASS`, every family `exit==0` / `ran_any_tests==true`.

| Family | Req | Crate | Named divergence (max abs_div) |
|--------|-----|-------|--------------------------------|
| nonsym_grow (Depthwise/Lossguide) | GPUT-18 | cb-backend | leaf-values 0.000e0 (bit-exact, bar 1e-4) |
| region_device | GPUT-18 | cb-backend | 0.000e0 (1/1) |
| exact_quantile + segmented_sort | GPUT-19 | cb-backend | 0.000e0 (10/10) |
| bootstrap_device | GPUT-09 | cb-backend | Bernoulli exact; Bayesian 2.384e-7 |
| mvs_device | GPUT-17 | cb-backend | 6.66e-16 … 4.44e-15 |
| ctr_device | GPUT-10 | cb-backend | good/total exact, value ≤1e-4 |
| device_nonsym_fit (e2e) | GPUT-18 | cb-train | full-fit pred parity (2/2) |
| device_region_fit (e2e) | GPUT-18 | cb-train | full-fit pred parity (1/1) |
| pairwise (deriv + batched Cholesky) | GPUT-11/21 | cb-backend | 0.000e0 (+ RV-13-04 oracle) |
| ranking (query grouping + det + stochastic) | GPUT-22 | cb-backend | der2 0.000e0 (+ RV-13-01/02/03 oracles) |
| multiclass (softmax der + multi-Newton) | GPUT-12 | cb-backend | 0.000e0 (9/9) |
| ordered (resident approx trajectory) | GPUT-13 | cb-backend | scan/partition_update 0.000e0 (bound 1e-9) |
| langevin (seeded Gaussian / SGLB) | GPUT-20 | cb-backend | 1.11e-16 … 4.44e-16 |

All named divergences are bit-exact except the inherently-stochastic bootstrap `2.384e-7`
and MVS `~1e-15`, all far under the ε=1e-4 device bar. This single aggregate row IS the
GPUT-14 sign-off (previously evidenced per-family only).

## Part B — Single-session BENCH-02 speed rows (HARD-02)

12 depth rows (depth-1 n=100k/300k/1M, depth-6 n=10k/100k/300k, × {depthwise, region}),
warm-run / JIT-excluded / queue-drained / median-of-3, 20 iters / 20 feat / 32 bins. Device
beats host CPU on every row; span **29.1× … 40.8×**; `bench_verdict: OK`; `depth6_ge20x:
true`. Crossover: depth-1 depthwise device first beats CPU at **n=100000** (recorded, not
gated, A4). Region `catboost_gpu_s = N/A` (no upstream Region grow_policy). Full table in
`bench/BENCH-03-SIGNOFF.md` and `bench/RESULTS.md`.

## Provenance (single session — D-08)

Tesla P100-PCIE-16GB, driver 580.159.04, CUDA release 12.8, seed 42,
`provenance.single_session: true`, date 2026-07-05. No mixed-session stitching — every
correctness verdict and speed cell comes from the one committed
`bench/phase15_cuda_oracle/result.json` (`734109a`).
