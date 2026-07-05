# Phase 15: Debt Discharge & CUDA Oracle Re-establishment - Context

**Gathered:** 2026-07-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Turn the v1.1 GPU evidence — which today is **stitched / per-family / `Pending`** — into a single authoritative Kaggle CUDA record, and close the four latent parity hazards, so every later parity/benchmark claim (Phases 19 GPU-infer, 21 adoption) rests on a trusted oracle.

**Delivers three things (ROADMAP §Phase 15, REQUIREMENTS HARD-01/02/03):**
- **HARD-01** — one aggregate GPUT-14 ε=1e-4 Kaggle CUDA correctness row covering **all** v1.1 device families as a single authoritative run.
- **HARD-02** — execute the missing **Phase-10 (depth-1)** and **Phase-11 (depth-6)** BENCH-02 speed rows on Kaggle CUDA; recompute the **BENCH-03** aggregate from real numbers with **no stitched Phase-12/13-only gaps**.
- **HARD-03** — resolve **RV-13-01..04** (each fixed with a demonstrating oracle).

**In scope:** Kaggle CUDA job execution (single kernel session), contained `cb-backend`/`cb-train` numeric fixes for RV-13-01..04, kernel/unit-level parity oracles for those fixes, BENCH-03 recompute, and requirement/gap bookkeeping.

**Out of scope (do NOT do in this phase):**
- **Wiring the e2e device-grow seam.** Families keep declining `Ok(None)`→CPU at the session level; the milestone defers e2e device grow. The hazard oracles are unit/kernel-level (see D-01), NOT a grow-loop wire-up.
- Re-baselining speed vs official CatBoost CPU — that is Phase 21's adoption benchmark.
- Any new device family / feature surface.
</domain>

<decisions>
## Implementation Decisions

### RV-13-01..04 hazard disposition (HARD-03)
- **D-01: Fix ALL FOUR with demonstrating oracles.** Not retire. Each hazard gets a real reproducing oracle, a fix, and a passing proof.
- **D-02: Oracles are unit/kernel-level direct invocations — NOT e2e device-grow.** This exercises each divergent path while staying inside the phase's "no grow-seam" boundary:
  - **RV-13-01** (`gpu_runtime/ranking.rs:766`) — invoke `descending_order_per_query` with **tied** perturbed values (`exp(approx)`+f32-Gumbel ties) and assert the decay-coefficient / order matches CPU's stable descending sort. The fix must make the device order stable-equivalent to CPU for ties (not merely rely on the frozen-fixture "no ties" assertion at :735).
  - **RV-13-02** (`gpu_runtime/ranking.rs:475`) — invoke `query_softmax_ders_host` on a **weighted** QuerySoftMax query whose max-approx doc has **weight ≤ 0**; seed the per-query exp shift from the **weight>0-only** max (mirroring CPU `TQuerySoftMaxError` / `ranking_der.rs:257-266`), assert der1/der2 parity ≤ 1e-4.
  - **RV-13-03** (`kernels/query_helper.rs:449`) — call `compute_group_means_host` with an all-empty-group offset (`q_offsets=[0,0]`, `n==0`); add the `n==0` short-circuit guard so no zero-length device buffer is launched (project HIP residency lesson); oracle = no fault + correct empty result on rocm in-env AND cuda.
  - **RV-13-04** (`gpu_runtime/pairwise.rs:1754`) — exercise the device f64 Cholesky solve path against the frozen wgpu host scorer on inputs with **near-equal borders**; ensure the host argmin tie-break agrees across wgpu and cpu/rocm/cuda (deterministic tie-break rule, not f64-accumulation-order-dependent).
- **D-03: The hazard-fix oracles ride in the SAME single Kaggle CUDA session as HARD-01** (numeric parity is authoritative only on Kaggle CUDA; ROCm in-env = non-gating smoke, but RV-13-03's fault-guard is also validated in-env).

### Aggregate GPUT-14 run assembly (HARD-01)
- **D-04: Single combined Kaggle CUDA kernel session.** One notebook runs every v1.1 device family's self-oracle + both BENCH-02 depth rows + the four RV-13 hazard oracles in **one kernel session** (one P100, one driver, one seed) and emits **one pass/fail verdict + one JSON**. This is what makes it "one authoritative row, no stitched gaps" — no `aggregate.py`-style multi-session stitching.
- **D-05: Correctness is a BLOCKING pre-gate to any timing** (carry the standing `bench/RESULTS.md` discipline: primitive+cindex bit-exact and depth-1 RMSE/Logloss ≤1e-5 must pass before any speed number is recorded; notebook halts on failure).

### BENCH-02 rows + BENCH-03 recompute (HARD-02)
- **D-06: Keep the pre-Phase-10 host-light CPU baseline** (`.planning/notes/gpu-training-host-light-root-cause.md`) for consistency with the existing 12 rows — just ADD the depth-1 and depth-6 device+CPU rows. Do NOT re-baseline here.
- **D-07: Depth-1/depth-6 speed rows run in the SAME kernel session as the correctness gate**, warm-run / JIT-excluded / lazy-CubeCL-queue-drained / median-of-N, depth/iterations/features/bins matching the existing protocol (depth per row, 20 iters, 20 features, 32 bins).
- **D-08: BENCH-03 recomputed from single-session provenance** — the recompute must NOT leave any cell sourced only from the Phase-12/13 sessions where a Phase-10/11 (depth-1/depth-6) row is now required. `catboost_gpu_s` stays informational (D-01 of the existing sign-off); Region stays `N/A` (no Region policy upstream).

### Evidence artifacts & bookkeeping (all three requirements)
- **D-09: Update `bench/BENCH-03-SIGNOFF.md` + `bench/RESULTS.md` in place** with the real un-stitched numbers (rewrite the "Standing debt" section to reflect discharge; fill the depth-1/depth-6 `TBD` oracle table with real session output — no fabrication).
- **D-10: Per-hazard fix/retire evidence lives in a phase artifact** (`15-EVIDENCE.md` or equivalent) — one entry per RV-13-0x: what diverged, the oracle, the fix, the passing result.
- **D-11: On completion, flip requirement/gap bookkeeping:** GPUT-14 → satisfied and HARD-01/02/03 → satisfied in `.planning/REQUIREMENTS.md`; clear the two `.planning/MILESTONES.md` "Known Gaps" entries (GPUT-14 aggregate + un-run BENCH-02) and the corresponding STATE.md Deferred Items.

### Claude's Discretion
- Exact notebook structure, JSON schema, and `aggregate.py` reuse vs replacement — as long as it produces one single-session authoritative record (D-04) with do-not-fabricate discipline.
- Precise deterministic tie-break rule for RV-13-01/04 (e.g., sort by f64 bits then stable index) — planner/researcher choose, oracle must prove CPU-equivalence.
- Whether median-of-N uses N=3 or N=5 — pick what fits the Kaggle time budget (Phase 14 combined run was ~15min; time limits are not a constraint here).
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & roadmap
- `.planning/REQUIREMENTS.md` §Debt & Hardening — HARD-01, HARD-02, HARD-03 (the locked requirements for this phase).
- `.planning/ROADMAP.md` §"Phase 15: Debt Discharge & CUDA Oracle Re-establishment" — goal, success criteria, milestone-wide context (Kaggle CUDA sole oracle; never add `cb-train` dep to `cb-backend`).
- `.planning/MILESTONES.md` §"Known Gaps" (v1.1) — the exact two rows (GPUT-14 aggregate, Phase-10/11 BENCH-02) this phase discharges.

### Existing GPU evidence to re-establish / rewrite in place
- `bench/BENCH-03-SIGNOFF.md` — current stitched Phase-12/13 sign-off; rewrite in place (D-09). Documents the host-light baseline, warm-run protocol, Region-N/A + border_count 128→32 divergences.
- `bench/RESULTS.md` — standing bench disciplines + the depth-1/depth-6 `TBD` oracle table to be filled with real numbers (D-09).
- `bench/phase12_cuda_oracle/bench02-result.json`, `bench/phase13_cuda_oracle/result.json`, `bench/phase14_cuda_signoff/` (incl. `aggregate.py`, `bench03-result.json`) — prior per-session sources + the aggregation script to reuse or supersede.
- `.planning/notes/gpu-training-host-light-root-cause.md` — the pre-Phase-10 host-light CPU baseline (D-06 keeps this baseline).

### RV-13 hazard source (recovered from archived Phase-13 review, git commit `0f457d9`)
- `crates/cb-backend/src/gpu_runtime/ranking.rs` (:766 RV-13-01, :475 RV-13-02) — descending sort tie-inversion; QuerySoftMax max-seed weight regime.
- `crates/cb-backend/src/kernels/query_helper.rs` (:449 RV-13-03) — `n==0` unguarded zero-length device buffer.
- `crates/cb-backend/src/gpu_runtime/pairwise.rs` (:1754 RV-13-04) — device f64 Cholesky vs frozen wgpu host scorer tie-break.
- CPU reference for RV-13-02: `ranking_der.rs` (`TQuerySoftMaxError`, ~:257-266, weight>0-only max seeding).

### Design docs (parity source of truth)
- `docs/CATBOOST_CUDA_KERNELS_DESIGN.md` — CUDA kernel parity reference for the device families under oracle.

### CubeCL (if any `#[cube]` kernel is touched by a fix)
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` — read before writing/editing kernel code.
- `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/cubecl_error_guideline.md` — read on ANY CubeCL build error before fixing.
- Landmine: `f32::MIN` sentinel (never `-inf` literal in a `#[cube]` kernel → HIP JIT reject); fixed-point-u64 deterministic reduction reused; always run rocm suite in-env after `#[cube]` changes.
</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Kaggle CUDA harness (proven Phases 12–14):** orchestrator drives the `kaggle` CLI itself; `git archive` tracked-source-only tarball (~1.8MB, NOT the 2.9G crates/ tree); background-poll `KernelWorkerStatus` until COMPLETE (~15min); auth via `~/.kaggle/access_token` (NOT `kaggle.json`). Verifier subagent cannot run GPU — orchestrator discharges the Kaggle run in-env.
- **`bench/phase14_cuda_signoff/aggregate.py`** — existing aggregation script; reuse or supersede for the single-session BENCH-03 recompute.
- **Per-family device self-oracles** (Phases 12/13) — 44 device self-oracle tests already exist under `--features cuda`; the aggregate GPUT-14 run collects them into one session.
- **fixed-point-u64 deterministic reduction + `f32::MIN` sentinel** — reuse for any determinism-sensitive fix.

### Established Patterns
- **`Ok(None)`→CPU fallback at session level** — the five device families decline to grow on-device today; RV-13-01..04 are latent because of this. The hazard oracles must invoke the kernels/helpers **directly** rather than through the (deferred) grow path.
- **Correctness-blocks-speed gate** — no speed number is valid unless the correctness oracles pass first (notebook halts on failure).
- **ROCm in-env = non-gating smoke; Kaggle CUDA = authoritative** — but RV-13-03's crash-guard fault is also validatable in-env.

### Integration Points
- `crates/cb-backend/src/gpu_runtime/` (ranking, pairwise) and `crates/cb-backend/src/kernels/query_helper.rs` — the four fix sites.
- Bench JSON + `bench/` markdown — the evidence surface updated in place.
- `.planning/REQUIREMENTS.md`, `.planning/MILESTONES.md`, `.planning/STATE.md` — the bookkeeping flip on completion.
</code_context>

<specifics>
## Specific Ideas

- "One authoritative row, no stitched gaps" is taken **literally**: a single Kaggle CUDA kernel session (one P100 / driver / seed) producing one verdict + one JSON — the opposite of the current BENCH-03 which explicitly aggregates multiple sessions with "mixed-session provenance."
- The RV-13-01..04 fixes are training-derivative / grow-path hazards. They are NOT exercised by Phase 19 (inference) or any other v1.2 phase's e2e path; the value of fixing now is a genuinely trustworthy oracle the moment a future milestone wires device-grow. Fix-and-oracle chosen over retire precisely to avoid carrying latent numeric debt into that future.
</specifics>

<deferred>
## Deferred Ideas

- **Wiring the e2e device-grow seam** (making RV-13-01/02/04 reachable through the real grow loop rather than direct kernel invocation) — deferred; the milestone keeps `Ok(None)`→CPU. A future milestone owns this.
- **Re-baseline speed vs official CatBoost CPU** — belongs to Phase 21 (Adoption / DX Capstone), not here.
- **RV-13-05..09** (dead-pattern coverage gate, multiclass memory blow-up, radix-sort overhead, Cholesky/PCG copy-paste consolidation) — efficiency/cleanup debt from the same Phase-13 review, NOT named in HARD-03; leave as hardening backlog unless a fix for 01/02/04 naturally subsumes one (e.g., RV-13-07 shares the RV-13-01 sort site).

None of the above expands Phase 15 scope — discussion stayed within the debt-discharge boundary.
</deferred>

---

*Phase: 15-debt-discharge-cuda-oracle-re-establishment*
*Context gathered: 2026-07-05*
