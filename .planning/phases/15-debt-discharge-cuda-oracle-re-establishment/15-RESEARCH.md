# Phase 15: Debt Discharge & CUDA Oracle Re-establishment - Research

**Researched:** 2026-07-05
**Domain:** GPU numeric parity (CubeCL device kernels), Kaggle CUDA oracle harness, bench aggregation, requirement/gap bookkeeping
**Confidence:** HIGH (all four fix sites read at exact `file:line`; harness + JSON schemas + bookkeeping surfaces confirmed in-repo)

## Summary

This is a **de-risking / debt-discharge** phase, not a feature phase. CONTEXT.md (D-01..D-11) already locks scope. The plan-blocking unknowns are: (1) the exact code shape of the four RV-13 fixes and their non-tautological oracles, (2) how to assemble ONE single-session Kaggle CUDA run (correctness gate + BENCH-02 depth rows + the four hazard oracles → one verdict + one JSON), and (3) the exact bookkeeping cells to flip. All three are now pinned to real evidence.

Two of the four hazards are genuine wrong-value bugs with a clear minimal fix: **RV-13-02** (device seeds the QuerySoftMax exp-shift from a max over **all** objects, while CPU seeds from max over **weight>0** objects — `ranking.rs:475` vs `ranking_der.rs:257-266`) and **RV-13-03** (`compute_group_means_host` guards `n_groups==0` but **not** `n==0`, so an all-empty-group offset launches a zero-length device buffer — `query_helper.rs:374-432`). The other two are **tie-break / determinism** hazards that are latent-but-plausibly-already-correct: **RV-13-01** (`descending_order_per_query`, `ranking.rs:739-769`) already uses a complemented-key stable LSD radix that *should* preserve tie order, and **RV-13-04** (`pairwise.rs:1754` device Cholesky vs frozen wgpu host scorer, argmax at `:1882-1898`) already re-resolves over exact host f64 — but neither has an oracle exercising the divergent (tied / near-equal-border) input, which is exactly the D-02 deliverable.

**Primary recommendation:** Structure the phase as (Wave A) four contained `cb-backend` numeric fixes each landing with its own direct unit/kernel oracle in the existing `*_test.rs` sibling files, ROCm-smoke in-env; (Wave B) a single new `bench/phase15_cuda_oracle/` harness that runs the 44 device self-oracles + both BENCH-02 depth rows + the four RV-13 oracles in one Kaggle CUDA kernel session emitting one verdict + one JSON; (Wave C) recompute BENCH-03 from that single-session JSON, rewrite `bench/BENCH-03-SIGNOFF.md` + `bench/RESULTS.md` in place, write `15-EVIDENCE.md`, and flip the requirement/gap bookkeeping. Correctness is a blocking pre-gate to any timing (D-05).

## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Fix ALL FOUR RV-13-01..04 with demonstrating oracles (not retire). Each gets a reproducing oracle, a fix, and a passing proof.
- **D-02:** Oracles are unit/kernel-level direct invocations — NOT e2e device-grow:
  - RV-13-01 (`ranking.rs:766`): invoke `descending_order_per_query` with **tied** perturbed values (`exp(approx)`+f32-Gumbel ties); assert decay-coefficient/order matches CPU stable descending sort. Fix must make device order stable-equivalent to CPU for ties (not rely on the frozen "no ties" assertion).
  - RV-13-02 (`ranking.rs:475`): invoke `query_softmax_ders_host` on a **weighted** query whose max-approx doc has **weight ≤ 0**; seed the per-query exp shift from the **weight>0-only** max (mirror CPU `TQuerySoftMaxError`/`ranking_der.rs:257-266`); assert der1/der2 parity ≤ 1e-4.
  - RV-13-03 (`query_helper.rs:449`): call `compute_group_means_host` with all-empty-group offset (`q_offsets=[0,0]`, `n==0`); add `n==0` short-circuit guard so no zero-length device buffer launches; oracle = no fault + correct empty result on rocm in-env AND cuda.
  - RV-13-04 (`pairwise.rs:1754`): exercise device f64 Cholesky vs frozen wgpu host scorer on **near-equal borders**; host argmin tie-break must agree across wgpu/cpu/rocm/cuda (deterministic rule, not f64-accumulation-order-dependent).
- **D-03:** Hazard-fix oracles ride in the SAME single Kaggle CUDA session as HARD-01 (Kaggle CUDA authoritative; ROCm in-env = non-gating smoke; RV-13-03's fault-guard also validated in-env).
- **D-04:** Single combined Kaggle CUDA kernel session — one notebook runs every v1.1 device family self-oracle + both BENCH-02 depth rows + the four RV-13 oracles in ONE kernel session (one P100, one driver, one seed) → one pass/fail verdict + one JSON. No `aggregate.py`-style multi-session stitching.
- **D-05:** Correctness is a BLOCKING pre-gate to any timing (notebook halts on failure).
- **D-06:** Keep the pre-Phase-10 host-light CPU baseline for consistency with the existing 12 rows — just ADD depth-1 and depth-6 device+CPU rows. Do NOT re-baseline.
- **D-07:** Depth-1/depth-6 speed rows run in the SAME kernel session as the correctness gate; warm-run / JIT-excluded / lazy-CubeCL-queue-drained / median-of-N; depth per row, 20 iters, 20 features, 32 bins.
- **D-08:** BENCH-03 recomputed from single-session provenance — no cell sourced only from Phase-12/13 where a Phase-10/11 depth row is now required. `catboost_gpu_s` stays informational; Region stays `N/A`.
- **D-09:** Update `bench/BENCH-03-SIGNOFF.md` + `bench/RESULTS.md` in place with real un-stitched numbers (rewrite "Standing debt" to reflect discharge; fill depth-1/depth-6 `TBD` oracle table with real output — no fabrication).
- **D-10:** Per-hazard fix/retire evidence in a phase artifact (`15-EVIDENCE.md`) — one entry per RV-13-0x: what diverged, the oracle, the fix, the passing result.
- **D-11:** On completion, flip bookkeeping: GPUT-14 → satisfied + HARD-01/02/03 → satisfied in `.planning/REQUIREMENTS.md`; clear the two `.planning/MILESTONES.md` "Known Gaps" (GPUT-14 aggregate + un-run BENCH-02); clear the corresponding STATE.md Deferred Items.

### Claude's Discretion

- Exact notebook structure, JSON schema, `aggregate.py` reuse vs replacement — as long as it produces one single-session authoritative record (D-04) with do-not-fabricate discipline.
- Precise deterministic tie-break rule for RV-13-01/04 (e.g., sort by f64 bits then stable index) — oracle must prove CPU-equivalence.
- Whether median-of-N uses N=3 or N=5 — pick what fits the Kaggle time budget (Phase 14 was ~15min; time limits are not a constraint).

### Deferred Ideas (OUT OF SCOPE)

- **Wiring the e2e device-grow seam** (making RV-13-01/02/04 reachable through the real grow loop) — families keep `Ok(None)`→CPU; a future milestone owns this.
- **Re-baseline speed vs official CatBoost CPU** — Phase 21 (Adoption / DX Capstone).
- **RV-13-05..09** (dead-pattern coverage gate, multiclass memory blow-up, radix-sort overhead, Cholesky/PCG consolidation) — not named in HARD-03; hardening backlog. EXCEPTION: RV-13-07 shares the RV-13-01 sort site, so an RV-13-01 fix may naturally touch it — do not expand scope to chase it.
- New device families / feature surfaces.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| HARD-01 | Aggregate ε=1e-4 Kaggle CUDA correctness sign-off (GPUT-14) across all v1.1 device families as one authoritative row | Single-session harness design (Architecture §2); 44 device self-oracles already exist under `--features cuda` (Phases 12/13); RV-13 oracles ride the same session (D-03) |
| HARD-02 | Phase-10 (depth-1) + Phase-11 (depth-6) BENCH-02 rows on Kaggle CUDA; BENCH-03 aggregate with real numbers (no stitched gaps) | BENCH-02 protocol confirmed (warm/JIT-excluded/queue-drained/median-of-N, 20 iters/20 feat/32 bins); depth-1 large-n caveat (D-10-09 escalation in `RESULTS.md`); BENCH-03 recompute path (Architecture §4) |
| HARD-03 | RV-13-01..04 latent parity hazards resolved with demonstrating oracles | All four fix sites read at exact `file:line`; per-hazard fix shape + oracle (Architecture §1) |

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| RV-13-01/02 ranking der fixes | `cb-backend` `gpu_runtime/ranking.rs` | `cb-compute/ranking_der.rs` (CPU ref, read-only) | Device der host-drivers live in cb-backend; CPU reference is transcribed/frozen, never a live dep (feature-unification landmine) |
| RV-13-03 group-means guard | `cb-backend` `kernels/query_helper.rs` | — | Pure host-side residency guard before kernel launch |
| RV-13-04 pairwise tie-break | `cb-backend` `gpu_runtime/pairwise.rs` | `cb-compute::calculate_pairwise_score` (frozen wgpu path) | Argmax over host-resident f64 scores; the two solve paths (device Cholesky vs host scorer) must agree on winner |
| Hazard oracles | `cb-backend` `*_test.rs` sibling files | ROCm in-env smoke | Source/test separation is MANDATORY; tests never in production `.rs` |
| Single-session CUDA run | `bench/phase15_cuda_oracle/` (new) | Kaggle CLI (orchestrator-driven) | Authoritative oracle of record; verifier subagent cannot run GPU |
| BENCH-03 recompute | `bench/*.md` + JSON | `aggregate.py` (supersede) | Evidence surface, updated in place (D-09) |
| Requirement/gap flip | `.planning/REQUIREMENTS.md`, `MILESTONES.md`, `STATE.md` | — | Bookkeeping on completion (D-11) |

## Project Constraints (from CLAUDE.md / AGENTS.md / ROADMAP)

- **`unwrap()` strictly prohibited in production** — hazard fixes use `thiserror` `CbError` variants; all `.unwrap()`/indexing lives in `*_test.rs`.
- **Source/test separation is MANDATORY** — no `#[cfg(test)] mod tests` in production `.rs`. Oracles go in sibling `*_test.rs` files (existing pattern: `ranking_stoch_test.rs`, `query_helper_test.rs`, `cholesky_solve_test`).
- **NEVER add a `cb-train` dependency to `cb-backend`** — feature-unification breaks the ROCm runtime. CPU references are FROZEN as literals (generated offline once from the independent `cb-train`/`cb-compute` path) so oracles stay non-tautological.
- **CubeCL `#[cube]` kernel rules** — read `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` before editing any kernel; on ANY build error read `cubecl_error_guideline.md` first. Landmine: `f32::MIN` sentinel, never `-inf` literal in a `#[cube]` body (HIP JIT reject, invisible to cpu/wgpu cargo check). **NOTE:** none of the four RV-13 fixes require editing a `#[cube]` kernel body (see per-hazard fix shapes below) — all are host-side driver/guard/tie-break changes. If a fix does touch a kernel, run the full ROCm suite in-env after.
- **Always run the ROCm suite in-env after any device-path change** (non-gating but catches HIP-only breakage).
- **Kaggle CUDA is the SOLE authoritative GPU oracle** (ROCm in-env = non-gating smoke). Orchestrator drives the `kaggle` CLI; verifier subagent cannot run GPU.
- **Do NOT fabricate numbers** — every bench cell comes from the actual run; `TBD` means not-yet-run.

## Architecture Patterns

### System Architecture Diagram

```
                        ┌─────────────────────────────────────────────┐
                        │  WAVE A — contained numeric fixes (in-env)   │
                        │                                              │
 RV-13-01 tie order ───▶│ ranking.rs::descending_order_per_query       │──┐
 RV-13-02 max seed  ───▶│ ranking.rs::query_softmax_ders_host          │  │
 RV-13-03 n==0 guard───▶│ query_helper.rs::compute_group_means_host    │  │ each fix +
 RV-13-04 tie-break ───▶│ pairwise.rs::select_best_split_over_scores   │  │ sibling *_test.rs
                        │                                              │  │ oracle
                        │  ROCm in-env smoke (non-gating)  ────────────┼──┘
                        └──────────────────────┬───────────────────────┘
                                               │ fixes committed
                                               ▼
                        ┌─────────────────────────────────────────────┐
                        │  WAVE B — ONE Kaggle CUDA kernel session      │
                        │  bench/phase15_cuda_oracle/oracle.py          │
                        │                                              │
                        │  Part A (BLOCKING pre-gate, D-05):           │
                        │    44 device self-oracles --features cuda    │
                        │    + 4 RV-13 oracles  ── ε=1e-4 ─────────────┤
                        │         │ ALL-PASS?                          │
                        │         ├── NO ──▶ emit verdict, sys.exit(2) │
                        │         │          (no speed number quoted)  │
                        │         ▼ YES                                │
                        │  Part B (timing): depth-1 + depth-6          │
                        │    BENCH-02 rows, warm/JIT-excluded/         │
                        │    queue-drained/median-of-N                 │
                        │         ▼                                    │
                        │  ONE verdict + ONE result.json               │
                        └──────────────────────┬───────────────────────┘
                                               │ single-session provenance
                                               ▼
                        ┌─────────────────────────────────────────────┐
                        │  WAVE C — evidence + bookkeeping (in place)   │
                        │  recompute BENCH-03 from the ONE json  ──────┤
                        │  rewrite BENCH-03-SIGNOFF.md + RESULTS.md     │
                        │  write 15-EVIDENCE.md (per-hazard)           │
                        │  flip REQUIREMENTS / MILESTONES / STATE (D-11)│
                        └─────────────────────────────────────────────┘
```

### Component Responsibilities

| File | Responsibility | Change |
|------|----------------|--------|
| `crates/cb-backend/src/gpu_runtime/ranking.rs` | RV-13-01 tie order (`:739-769`), RV-13-02 max seed (`:452-505`) | edit |
| `crates/cb-backend/src/kernels/query_helper.rs` | RV-13-03 `n==0` guard (`:374-432`) | edit |
| `crates/cb-backend/src/gpu_runtime/pairwise.rs` | RV-13-04 tie-break (`:1807-1910`) | edit |
| `crates/cb-backend/src/gpu_runtime/ranking_stoch_test.rs` | RV-13-01/02 oracles | add tests |
| `crates/cb-backend/src/kernels/query_helper_test.rs` | RV-13-03 oracle (extends `zero_weight_query_mean_is_zero`) | add test |
| `crates/cb-backend/src/kernels/cholesky_solve_test.rs` (or `pairwise_*_test.rs`) | RV-13-04 oracle | add test |
| `bench/phase15_cuda_oracle/oracle.py` (new) | single-session runner | create |
| `bench/phase15_cuda_oracle/result.json` (new) | one verdict + one JSON | create (from run) |
| `bench/BENCH-03-SIGNOFF.md`, `bench/RESULTS.md` | evidence, in place | rewrite |
| `.planning/phases/15-.../15-EVIDENCE.md` (new) | per-hazard evidence | create |
| `.planning/REQUIREMENTS.md`, `MILESTONES.md`, `STATE.md` | bookkeeping flip | edit |

### Pattern 1: Non-tautological frozen-CPU-reference oracle

**What:** The device der driver lives in `cb-backend`; the CPU reference is FROZEN as literals in the `*_test.rs`, generated once offline from the *independent* `cb-train`/`cb-compute` implementation. This keeps the oracle honest (different code paths) without a `cb-train` dep.
**When to use:** every RV-13 oracle.
**Example (existing pattern, `ranking_stoch_test.rs`):**
```rust
// Source: crates/cb-backend/src/gpu_runtime/ranking_stoch_test.rs
#![cfg(not(feature = "wgpu"))]
const TOL: f64 = 1e-4; // D-07 GPU bar, looser than the CPU ref's own ≤1e-5
fn device_backend_active() -> bool { cfg!(any(feature = "rocm", feature = "cuda")) }
// ε assertion SKIPS off rocm/cuda (record-only) so a default `cpu` run is not a
// CPU-vs-CPU false-pass (WR-01 anti-false-pass).
```

### Pattern 2: Correctness-blocks-speed single-session harness

**What:** `oracle.py` Part A runs correctness under `--features cuda` and `sys.exit(2)` BEFORE any timing if a family fails; Part B times only if Part A is ALL-PASS. One `result.json`.
**When to use:** the Wave B notebook.
**Example (existing pattern, `phase14_cuda_signoff/oracle.py`):** Part A per-family self-oracle gate → `sys.exit(2)` on SOME-FAIL before Part C timing; compact JSON to `/kaggle/working`; build in `/tmp` so Kaggle output stays tiny.

### Anti-Patterns to Avoid

- **`aggregate.py`-style multi-session stitching** — the whole point of HARD-01/D-04 is ONE session. Supersede `aggregate.py`; do not extend it to read Phase-10/11 as separate files.
- **Fabricating a Region `catboost_gpu_s`** — Region has no upstream policy; stays `N/A` (Pitfall 4 in the existing sign-off), never a proxy.
- **Recording a speed number when correctness failed** — D-05 blocking gate; notebook halts.
- **A CPU-vs-CPU false pass** — on the default `cpu` backend the "device" IS the host; numeric asserts must skip off real device (`device_backend_active()`), record-only.
- **Editing a `#[cube]` kernel body without reading the CubeCL manual** — and never a `-inf` literal.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Descending per-query sort | A second sort algorithm | `segmented_radix_sort` (`exact_quantile.rs:162`) already reused at `ranking.rs:760,766` | Stable LSD radix; the acceptance-criterion sort reuse |
| Stable tie order | Manual index-carry loop | The radix already carries the permutation vector (idx0) through both passes | Composition of stable passes is a stable full sort |
| BENCH-02 workload | New synthetic generator | `gen()` (numpy repro of `cb-train/tests/bench_grow_speed_test.rs::gen()`) in `phase14_cuda_signoff/oracle.py` | Frozen config: integer-binned 0..31 f32 cols, 20 feat, 32 bins, seed 42 |
| BENCH-03 verdict | New pass/fail logic | The `>=20x` gate + verdict shape from `aggregate.py` (`GE20X_GATE=20.0`) | Established discipline; recompute reads the ONE json instead of two |
| CatBoost-GPU timing arm | New comparison harness | Part C of `phase14_cuda_signoff/oracle.py` (informational, Region N/A, border_count 32, quantization-cost caveat) | Already handles the four documented divergences |

**Key insight:** almost everything this phase needs already exists (the harness, the CPU refs, the self-oracles, the sort, the workload gen). The novel work is (a) four small numeric fixes + oracles, (b) collapsing the run into ONE session, (c) truthful in-place evidence rewrite. Low code-change risk, high de-risking value.

## Common Pitfalls

### Pitfall 1: Assuming RV-13-01 is broken (it is plausibly already tie-correct)
**What goes wrong:** planner writes a "fix" that changes working code and risks regressing the passing frozen-seed oracle.
**Why it happens:** the review flagged RV-13-01 as a *latent hazard*, not a confirmed wrong value. The current `descending_order_per_query` (`ranking.rs:739-769`) already complements the radix key (`!v.to_bits()`) and runs two **stable** LSD passes (`segmented_radix_sort` → `run_radix_sort_device` = `radix_bit_flag_kernel` → `full_scan` onesBefore → `reorder_one_bit_scatter`, each pass stable per `exact_quantile.rs:62-64`). By construction that preserves ties in original index order — matching upstream's stable descending sort.
**How to avoid:** the D-01 deliverable is the *oracle with actual ties*. Write the tie oracle FIRST. If it passes as-is, the "fix" is confirmatory (add a doc-anchored `debug_assert`/comment + the oracle) and RV-13-04-style defensive index-carry is the recommended default only IF the oracle reveals a flip. Record the outcome honestly in `15-EVIDENCE.md`.
**Warning signs:** an oracle that passes before any code change means the hazard was already mitigated — that is a valid HARD-03 discharge (fix = "verified stable + oracle added"), not a reason to churn code.

### Pitfall 2: RV-13-02 fix must filter the MAX by weight>0, not just document it
**What goes wrong:** leaving the `compute_group_max_host(approx, ...)` call (max over ALL objects) and only editing the comment — the weighted-query-with-negative-weight-max case stays wrong.
**Why it happens:** the current comment at `ranking.rs:473-474` explicitly says "Uniform-weight covered regime: max over all objects == the CPU max-over-(weight>0) seed" — true only for uniform weights. CPU `ranking_der.rs:257-266` seeds `max_approx` scanning `if w > 0.0` only.
**How to avoid:** compute the per-query max over weight>0 objects (a weight-aware group-max: either a new host helper mirroring CPU, or filter `approx`/pass the weight column into a max that ignores w≤0). The empty-weight-group case falls back to the `f64::MIN` seed but the downstream `sum_weighted_targets > 0` guard short-circuits before exp (mirror CPU exactly). Oracle: a weighted query where the global-max doc has weight ≤ 0 must give der ≤ 1e-4 vs the frozen CPU der.
**Warning signs:** the oracle passes with a uniform-weight fixture but you never added a w≤0-on-the-max-doc case — that is the tautology to avoid.

### Pitfall 3: RV-13-03 guard must return the RIGHT empty result, not just avoid the fault
**What goes wrong:** adding `if n==0 { return Ok(Vec::new()) }` when `n_groups > 0` — wrong length (callers expect `n_groups` group means).
**Why it happens:** the existing guard only covers `n_groups==0` (`query_helper.rs:381-383`). With `q_offsets=[0,0]`, `n_groups==1` and `n==0`: an empty group's mean is `0.0` (CPU `queryAvrg 0` when denominator ≤ 0, `ranking_der.rs:231`).
**How to avoid:** short-circuit `if n == 0 { return Ok(vec![0.0; n_groups]); }` (all-empty groups → zero means), placed before `client.create` so no zero-length device buffer is bound (project HIP residency lesson: never create/read a 0-len handle). Also audit the two sibling helpers with the same shape: `compute_group_max_host` (`:436-463`) and `remove_group_means_host` (`:467-507`, already guards `n==0` at `:473`) — apply the same guard to `compute_group_max_host` if the oracle can reach it, but stay inside the RV-13-03 boundary (the named site is `compute_group_means_host`).
**Warning signs:** oracle only checks "no panic" but not the returned value/length.

### Pitfall 4: RV-13-04 — the scores differ by backend, so exact-equality tie-break rarely fires
**What goes wrong:** trusting the current `score == best_score` exact-f64 tie-break (`pairwise.rs:1893`) to make wgpu (host scorer `calculate_pairwise_score`, `:1768`) and cuda (device Cholesky, `:1754`) pick the same border. On near-equal borders the two solve paths produce f64 scores differing by ~1e-13 accumulation noise, so the winner is decided by whichever is fractionally larger — which flips between backends. Exact `==` almost never triggers.
**Why it happens:** the device Cholesky path and the frozen host scorer are *different* float accumulation orders. The self-oracle `cholesky_solve_test` claims bit-for-bit, but that is not guaranteed on near-degenerate SPD systems with near-equal borders.
**How to avoid (recommended default — ambiguity flagged below):** make the argmax tie-break *near-equal tolerant and index-deterministic*: treat two candidate scores as tied when `|a - b| <= tol * max(|a|, |b|, 1.0)` and break by lowest candidate index (the strict first-wins parity contract == `select_best_candidate`). The oracle constructs inputs with two borders whose true scores are within that band and asserts the device-Cholesky path and the frozen wgpu host-scorer path select the SAME `BestSplit`. `tol` sized just above the observed device-vs-host score delta (start ~1e-9 relative; the oracle settles it).
**Warning signs:** an oracle that only uses well-separated borders never exercises the flip.

### Pitfall 5: Depth-1 device ≥ CPU is only achievable at LARGE n (physics, not tuning)
**What goes wrong:** running the depth-1 BENCH-02 row at small n (10k) and recording a device-slower-than-CPU "failure."
**Why it happens:** the pinned D-10-09 escalation in `bench/RESULTS.md:19-36` — a depth-1 stump is the most launch-overhead-bound workload in the milestone; device wins only at large n (~1e5–1e6+) where the `O(n·features)` histogram amortizes fixed launch latency.
**How to avoid:** run the depth-1 row on the large-n `SPEED_CONFIG` (~1e6×50, tunable above break-even). If device still loses at large n, record the measured crossover (or its absence) and surface it — do not silently pass/fail. Depth-6 (Phase 11) is where device dominance is unambiguous (existing 12 rows are all depth-6, 23.9–42.1×).
**Warning signs:** a depth-1 device number recorded at n=10k without the large-n crossover note.

### Pitfall 6: Line numbers in CONTEXT are from archived commit `0f457d9`, not HEAD
**What goes wrong:** editing the wrong function because `query_helper.rs:449` (CONTEXT) now points inside `compute_group_max_host` at HEAD, not `compute_group_means_host`.
**Why it happens:** RV-13 sites were recovered from the archived Phase-13 review at git `0f457d9`; the file has since shifted. The **named function** is authoritative, not the line number. At HEAD: `compute_group_means_host` is `query_helper.rs:374-432`; `descending_order_per_query` is `ranking.rs:739-769`; `query_softmax_ders_host` is `ranking.rs:452-505`; the pairwise device-Cholesky call is `pairwise.rs:1754` with the argmax at `:1882-1898`.
**How to avoid:** anchor every edit on the function name (verified at HEAD in this research), then confirm the line.

## Code Examples

### RV-13-02: the CPU weight>0 max-seed to mirror
```rust
// Source: crates/cb-compute/src/ranking_der.rs:257-266 (CPU reference — the parity target)
let mut max_approx = f64::MIN;               // upstream -numeric_limits::max()
for i in 0..group.size() {
    let w = weight_at(i);
    let a = approx_slice.get(i).copied().unwrap_or(0.0);
    if w > 0.0 {                             // <-- weight>0-ONLY seed (the fix target)
        if a > max_approx { max_approx = a; }
        // ...
    }
}
// device query_softmax_ders_host currently seeds from compute_group_max_host(approx) — max over
// ALL objects (ranking.rs:475). Fix: seed from a weight>0-aware group max.
```

### RV-13-03: the guard shape
```rust
// Source: crates/cb-backend/src/kernels/query_helper.rs:374-432 (compute_group_means_host)
let n = values.len();
let n_groups = q_offsets.len().saturating_sub(1);
if n_groups == 0 { return Ok(Vec::new()); }  // existing guard
// PROPOSED FIX (before client.create — no zero-length device buffer, HIP residency lesson):
if n == 0 { return Ok(vec![0.0; n_groups]); } // all-empty groups → zero means (CPU queryAvrg 0)
```

### RV-13-04: current argmax (the tie-break to make backend-agnostic)
```rust
// Source: crates/cb-backend/src/gpu_runtime/pairwise.rs:1882-1898
let mut best_score = f64::NEG_INFINITY;
let mut best_c = u32::MAX;
for &cand in best_idxs.iter() {
    let score = match scores.get(cand as usize) { Some(&s) => s, None => continue };
    // CURRENT: exact f64 tie-break — rarely fires across device-Cholesky vs host-scorer scores.
    let take = score > best_score || (score == best_score && cand < best_c);
    // PROPOSED: near-equal band + lowest-index (deterministic, not accumulation-order-dependent).
    if take { best_score = score; best_c = cand; }
}
```

## Runtime State Inventory

> Not a rename/refactor phase — no stored data/service-config/OS-registration surface. The only "state" is documentation/bookkeeping (below) and the numeric fixes are code-only. Included for completeness.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — no datastore keys reference RV-13 or GPUT-14 | none |
| Live service config | Kaggle kernel (`yensen2/catboost-rs-phase15-*`) created at run time; not git-tracked | orchestrator creates via `kaggle` CLI |
| OS-registered state | None | none |
| Secrets/env vars | Kaggle auth via `~/.kaggle/access_token` (NOT `kaggle.json`) — unchanged | none |
| Build artifacts | Kaggle build lives in `/tmp` (Phase-12/13/14 pattern) so output stays tiny (~1.8MB `git archive` tracked-source tarball, not the 2.9G crates/) | reuse pattern |
| Bookkeeping (doc state) | REQUIREMENTS traceability + inline checkboxes; MILESTONES Known Gaps; STATE Deferred Items | flip on completion (D-11 — see Bookkeeping Surface below) |

## Bookkeeping Surface (D-11 — exact locations)

| File | Cell / line | Current | Flip to |
|------|-------------|---------|---------|
| `.planning/REQUIREMENTS.md` | `:16-18` inline checkboxes HARD-01/02/03 | `[ ]` | `[x]` |
| `.planning/REQUIREMENTS.md` | `:80-82` traceability table HARD-01/02/03 | `Pending` | `Complete` |
| `.planning/MILESTONES.md` | `:21` Known Gap — GPUT-14 aggregate | "status **Pending** ... never run as a single aggregate" | discharged (Phase 15) |
| `.planning/MILESTONES.md` | `:22` Known Gap — Phase-10/11 BENCH-02 | "never executed ... stitches Phase-12/13 only" | discharged (Phase 15) |
| `.planning/STATE.md` | `:501` Deferred — GPUT-14 requirement | "pending — ... not executed" | resolved (Phase 15) |
| `.planning/STATE.md` | `:502-503` Deferred — Phase-10/11 BENCH-02 rows | "not executed" | resolved (Phase 15) |
| `.planning/STATE.md` | `:504-505` (optional) Phase-10 verification/UAT depth-1 Kaggle gate | human_needed / 2 pending | resolved if the depth-1 row runs |

**Note on GPUT-14 in REQUIREMENTS.md:** GPUT-14 is a v1.1 requirement; at HEAD it is tracked in STATE Deferred Items (`:501`) and MILESTONES Known Gaps (`:21`), NOT as a live checkbox in the v1.2 `REQUIREMENTS.md` (which lists HARD-01..03, the v1.2 wrappers). D-11's "GPUT-14 → satisfied" is discharged by clearing the STATE/MILESTONES entries and marking HARD-01 complete. Verify no other `GPUT-14 Pending` string lingers via `grep -rn "GPUT-14" .planning/`.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| BENCH-03 aggregates 2 committed sessions (`aggregate.py`, mixed-session provenance) | ONE Kaggle CUDA kernel session → one verdict + one JSON (D-04) | this phase | supersede `aggregate.py`; recompute reads the single json |
| GPUT-14 evidenced per-family, no aggregate row | single aggregate ε=1e-4 row covering all families (HARD-01) | this phase | flips GPUT-14 Pending → satisfied |
| BENCH-02 depth-1/depth-6 rows `TBD` in `RESULTS.md` | real numbers from the single session (HARD-02) | this phase | no stitched gaps |
| RV-13-01..04 latent (unreached via `Ok(None)`→CPU) | fixed + direct unit/kernel oracles (HARD-03) | this phase | trustworthy oracle for future device-grow wiring |

**Deprecated/outdated:**
- `bench/phase14_cuda_signoff/aggregate.py` two-source stitching — superseded by the single-session recompute (keep the `>=20x` verdict logic; drop the two-file `load_rows`).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | RV-13-01's current complemented-key stable radix already preserves tie order (so the "fix" may be confirmatory + oracle) | Pitfall 1 | LOW — the oracle settles it; if it flips, add a defensive tertiary index key. Either outcome discharges HARD-03 with evidence. |
| A2 | RV-13-04's correct fix is a near-equal-tolerant, lowest-index tie-break (rather than forcing device Cholesky to bit-match the host scorer) | Pitfall 4 | MEDIUM — CONTEXT D-02 says "deterministic tie-break rule," favoring this reading. Alternative (make solves bit-identical) is larger scope. Oracle proves CPU-equivalence either way; `tol` value is discretion. |
| A3 | The empty-group mean for RV-13-03 is `0.0` per group (CPU `queryAvrg 0` when denominator ≤ 0) | Pitfall 3 | LOW — confirmed at `ranking_der.rs:231`; existing `zero_weight_query_mean_is_zero` test asserts `ref_means[0]==0.0`. |
| A4 | Depth-1 BENCH-02 may legitimately NOT beat CPU even at large n; the phase records the crossover rather than forcing a pass | Pitfall 5 | MEDIUM — D-10-09 escalation says the Kaggle run is the arbiter; success criterion is "row executed + recorded," not "device wins." Planner should not gate HARD-02 on depth-1 device≥CPU. |
| A5 | median-of-N and single-session assembly fit one Kaggle run (~15min like Phase 14) | Architecture §3 | LOW — CONTEXT says time is not a constraint; N=3 or N=5 is discretion. |
| A6 | No RV-13 fix requires editing a `#[cube]` kernel body (all host-side driver/guard/tie-break) | Project Constraints | LOW — verified by reading each site; if RV-13-02's weight-aware max needs a new kernel, apply CubeCL rules + `f32::MIN` + ROCm-in-env re-run. |

## Open Questions (RESOLVED)

1. **Is `run_radix_sort_device`'s single-segment scatter provably stable for equal keys?**
   - What we know: doc claims "each pass is stable, so the composition is a stable full sort" (`exact_quantile.rs:62-64`); it is a 1-bit split scatter (`reorder_one_bit_scatter`) which is the textbook stable radix primitive.
   - What's unclear: whether the device scatter is bit-stable under GPU lane races for equal keys (should be — it's a prefix-sum-positioned scatter, not an atomic race).
   - Recommendation: the RV-13-01 tie oracle empirically settles this on cuda/rocm. If stable, done; if not, carry the original index as a tertiary radix key.
   - **RESOLVED: oracle-settled — 15-01 Task 1 writes the tie oracle first (confirm-or-fix per A1); a tertiary original-index radix-key fallback branch is planned if the oracle shows instability. "Verified stable + oracle added" is a valid HARD-03 discharge (D-01).**

2. **Does RV-13-02's weight-aware group-max need a new kernel or a host-side filter?**
   - What we know: `compute_group_max_host` ignores weights; CPU filters `w>0`.
   - What's unclear: cheapest correct shape — (a) a host loop computing weight>0 per-query max (no kernel), or (b) a new weight-aware `compute_group_max_weighted` kernel.
   - Recommendation: prefer (a) host-side (the max-seed is O(n), already read back to host as `group_max`; no residency benefit to a kernel) — avoids a `#[cube]` edit. Oracle is identical either way.
   - **RESOLVED: host-side weight>0 filter — 15-01 Task 2 adopts option (a), no `#[cube]` edit, mirroring CPU `ranking_der.rs:257-266`.**

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Kaggle CUDA (Tesla P100) | HARD-01/02 authoritative oracle | ✓ (orchestrator-driven `kaggle` CLI) | P100, CUDA 12.8, driver 580.x | none — sole authoritative GPU oracle |
| `kaggle` CLI + `~/.kaggle/access_token` | dispatch the run | ✓ (proven Phases 12–14) | — | none |
| ROCm gfx1100 in-env | non-gating smoke + RV-13-03 fault-guard validation | ✓ | ROCm 7.1, wave32 | — (non-gating) |
| official `catboost` (GPU) | informational `catboost_gpu_s` column | ✓ (Kaggle image) | 1.2.10, task_type='GPU' | Region N/A; column is informational only |
| `--features cuda` build | 44 device self-oracles | ✓ (builds in `/tmp` on Kaggle) | — | none |

**Missing dependencies with no fallback:** none — the harness is proven across Phases 12–14.
**Missing dependencies with fallback:** ROCm `Atomic<u64>` regression in-env means depth>1 resident grow cannot run on rocm; that is why Kaggle CUDA is the oracle of record (in-env is smoke). Not a blocker for this phase's fixes (host-side).

## Validation Architecture

> `.planning/config.json` not present with `nyquist_validation: false`; treating as enabled. In this phase the oracles ARE the validation.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` (sibling `*_test.rs`), `approx` for float asserts |
| Config file | none — cargo test |
| Quick run command | `cargo test -p cb-backend --no-default-features --features cpu <test_name>` (compiles, CPU-skips the ε assert) |
| Full suite command | `cargo test -p cb-backend --no-default-features --features rocm` (in-env smoke) + the Kaggle `--features cuda` run (authoritative) |

### Phase Requirements → Test Map
| Req | Behavior | Test Type | Automated Command | File Exists? |
|-----|----------|-----------|-------------------|-------------|
| HARD-03 RV-13-01 | tied perturbed values → device order == CPU stable descending | unit/kernel | `cargo test -p cb-backend --features cuda tie_order` | ❌ Wave A (add to `ranking_stoch_test.rs`) |
| HARD-03 RV-13-02 | weighted query, max-doc w≤0 → der ≤1e-4 vs frozen CPU | unit/kernel | `cargo test -p cb-backend --features cuda softmax_weight_max_seed` | ❌ Wave A (add to `ranking_stoch_test.rs`) |
| HARD-03 RV-13-03 | `q_offsets=[0,0]`, n==0 → no fault + `[0.0]` | unit/kernel | `cargo test -p cb-backend --features rocm empty_group_means` | ⚠️ extends `query_helper_test.rs::zero_weight_query_mean_is_zero` |
| HARD-03 RV-13-04 | near-equal borders → device-Cholesky == wgpu-host-scorer winner | unit/kernel | `cargo test -p cb-backend --features cuda pairwise_near_equal_tiebreak` | ❌ Wave A (add to `cholesky_solve_test`/`pairwise_*_test`) |
| HARD-01 | 44 device self-oracles + 4 RV-13 oracles ALL-PASS ε=1e-4, one session | integration | Kaggle `oracle.py` Part A | ❌ Wave B |
| HARD-02 | depth-1 + depth-6 BENCH-02 rows, one session | integration | Kaggle `oracle.py` Part B | ❌ Wave B |

### Sampling Rate
- **Per task commit:** `cargo test -p cb-backend --no-default-features --features cpu` (compile + CPU-skip) then `--features rocm` in-env smoke.
- **Per wave merge:** full `cb-backend --features rocm` suite in-env.
- **Phase gate:** the single Kaggle CUDA session ALL-PASS (correctness) before any speed number, then BENCH-03 recompute.

### Wave 0 Gaps
- [ ] RV-13-01 tie oracle in `ranking_stoch_test.rs`
- [ ] RV-13-02 weight>0-max-seed oracle in `ranking_stoch_test.rs`
- [ ] RV-13-03 empty-group oracle extending `query_helper_test.rs`
- [ ] RV-13-04 near-equal-border oracle in the pairwise/cholesky test
- [ ] `bench/phase15_cuda_oracle/oracle.py` single-session runner (Part A gate + Part B timing)

## Package Legitimacy Audit

**N/A — this phase installs no new external packages.** All work is contained numeric edits in `cb-backend` (existing deps: `cubecl`, `cb-core`, `cb-compute`, `thiserror`) + a Python harness (stdlib + numpy + the already-present `catboost==1.2.10` on the Kaggle image). No `Cargo.toml` / `requirements` additions. No SLOP/SUS surface.

## Security Domain

**N/A for enforcement** — code-only numeric parity fixes and a benchmark harness; no auth, input-validation boundary, cryptography, session, or access-control surface is introduced. The one residency-safety item (RV-13-03 zero-length device buffer guard) is a robustness fix, not a security control. No ASVS category applies.

## Sources

### Primary (HIGH confidence — read at exact file:line this session)
- `crates/cb-backend/src/gpu_runtime/ranking.rs` `:452-505` (RV-13-02), `:739-769` (RV-13-01 descending sort), `:807-924` (yetirank der core / call site)
- `crates/cb-backend/src/kernels/query_helper.rs` `:374-432` (RV-13-03 `compute_group_means_host`), `:436-463` (`compute_group_max_host`)
- `crates/cb-backend/src/gpu_runtime/pairwise.rs` `:1690-1799` (device Cholesky path), `:1807-1910` (argmax tie-break)
- `crates/cb-compute/src/ranking_der.rs` `:230-299` (CPU QuerySoftMax weight>0 max seed — RV-13-02 target)
- `crates/cb-backend/src/kernels/exact_quantile.rs` `:155-212` (`segmented_radix_sort` stability), `:62-64` (per-pass stability claim)
- `crates/cb-backend/src/gpu_runtime/ranking_stoch_test.rs`, `crates/cb-backend/src/kernels/query_helper_test.rs` (oracle patterns)
- `bench/BENCH-03-SIGNOFF.md`, `bench/RESULTS.md` (standing disciplines, 12-row matrix, TBD tables, D-10-09 escalation)
- `bench/phase14_cuda_signoff/aggregate.py` + `oracle.py`, `bench/phase12_cuda_oracle/bench02-result.json`, `bench/phase13_cuda_oracle/result.json`, `bench/phase14_cuda_signoff/bench03-result.json` (JSON schemas, session structure)
- `.planning/REQUIREMENTS.md` `:16-18,80-82`; `.planning/MILESTONES.md` `:19-24`; `.planning/STATE.md` `:479-509`; `.planning/ROADMAP.md` `:60-69` (bookkeeping + success criteria)
- `.planning/phases/15-.../15-CONTEXT.md` (D-01..D-11)

### Secondary (MEDIUM confidence)
- MEMORY.md phase 12/13/14 outcome notes (Kaggle CLI drive, `Ok(None)`→CPU, RV-13 provenance from git `0f457d9`, ROCm Atomic<u64> regression)

### Tertiary (LOW confidence)
- none — all claims anchored in-repo.

## Metadata

**Confidence breakdown:**
- Fix sites & shapes: HIGH — all four read at exact HEAD line numbers; CPU references confirmed.
- Single-session harness: HIGH — cloned from proven Phase-14 `oracle.py` structure; JSON schemas confirmed.
- RV-13-01 tie-correctness (already-mitigated?): MEDIUM — strong structural argument (stable radix), oracle settles it.
- RV-13-04 tie-break rule: MEDIUM — recommended default with flagged ambiguity (A2); oracle proves CPU-equivalence.
- Bookkeeping surface: HIGH — exact files/lines located.

**Research date:** 2026-07-05
**Valid until:** 2026-08-04 (stable — internal repo state; re-verify HEAD line numbers if `cb-backend` changes before planning)
