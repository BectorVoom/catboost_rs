# Phase 14: Comprehensive Kaggle CUDA Speed Benchmark + Parity Sign-Off - Context

**Gathered:** 2026-07-04
**Status:** Ready for planning

<domain>
## Phase Boundary

The **terminal phase of the v1.1 GPU-performance milestone**. Its sole requirement is **BENCH-03**: a comprehensive, milestone-closing speed-parity **sign-off** that *aggregates* the per-phase BENCH-02 speed checks already recorded in Phases 10–13 into one head-to-head document, with CUDA correctness re-confirmed as a **blocking pre-flight gate** before any speed number is trusted.

This phase does **NOT** measure speed for the first time and does **NOT** extend device coverage. Phases 10–13 each carried their own standing BENCH-02 Kaggle CUDA speed check (Phase 12/13 committed real P100 `result.json` numbers at 30–42× device≫host-CPU). Phase 14 rolls those up, adds the missing official-CatBoost-GPU comparison timing, and produces the signed-off "the device-resident path closes the >20× gap" deliverable that ends the milestone.

**Scope anchor — LOCKED, carried forward from Phases 10–13 (not re-decided here):**
- **ε bar:** device path holds **ε=1e-4 vs the Rust CPU path** (≤1e-5 for the depth-1 whole-dataset tree); CPU/host path stays oracle-locked ≤1e-5 and **byte-unchanged** (D-04 no-regression).
- **Kaggle CUDA is the sole correctness+speed authority** — a human-gated `--features cuda` notebook, reusing the Phase-10 `bench/` harness. ROCm in-env is an optional compile/smoke convenience, **not a gate**.
- **Measurement protocol:** warm-run / JIT-excluded / train-only wall-clock; drain the lazy CubeCL queue (read-back/predict) before stopping the clock; verify CUDA backend active via `nvidia-smi`; warm one untimed fit; re-run the oracle before timing.
- **Standing landmines:** never add a `cb-train` dep to `cb-backend` (transcribe CPU refs inline); no `-inf` float literals in `#[cube]` kernels (`f32::MIN` sentinel); deterministic reduction mandatory; never read a `Handle` through a foreign client.
- **No new compute crates.** `criterion 0.7.x` (dev-dep) permitted for optional in-env ROCm relative-timing regression during development; optional `profile-tracy`/`tracing` behind a `profiling` Cargo feature.

**Observed state at discussion time (drives D-04 below):** `bench/RESULTS.md`'s depth-1/depth-6 correctness+speed table is still all `TBD`, and **GPUT-14 is still marked Pending** in REQUIREMENTS.md — even though Phases 12/13 have committed real-P100 `result.json` files. The user decided (D-04) this phase does NOT formally own that backfill.

</domain>

<decisions>
## Implementation Decisions

### BENCH-03 sign-off criterion (what "closes the >20× gap" must show)
- **D-01 (hard gate = ≥20× vs host-CPU baseline; CatBoost-GPU informational):** The **only blocking** BENCH-03 pass criterion is that the device-resident training path beats the **pre-Phase-10 host-light CPU baseline by ≥20×** across the workload matrix — i.e. the original >20× *slowdown* (device grew trees host-light, ~20× slower than CPU; see `gpu-training-host-light-root-cause.md`) is demonstrably **reversed**. Phase 12 already recorded 30–42× device≫host-CPU, so this is expected to pass on aggregation. The **head-to-head vs official CatBoost GPU is recorded and discussed as INFORMATIONAL context only** — any shortfall (catboost-rs slower than CatBoost's hand-tuned CUDA) is **documented, not a milestone blocker**. Rationale: the milestone goal is "close our own >20× host-light gap," which is fully in our control; matching a mature C++/CUDA library's absolute throughput is a stretch goal, not the definition of done.

### Workload matrix (datasets & configs)
- **D-02 (synthetic large-n only — no external dataset staging):** The comprehensive matrix reuses the existing `bench/generator.py` **large-n synthetic** workloads (the D-06 `SPEED_CONFIG`, ~1e5–1e6 rows, above the depth-1 break-even), spanning the loss families already timed per-phase (at minimum depth-6 RMSE + Logloss; extend to a ranking + multiclass config where a per-phase BENCH-02 number exists to aggregate). **No Higgs/Epsilon or other external named datasets** — avoids Kaggle data-staging overhead and keeps the run fully reproducible. Trade-off accepted: synthetic random data makes the timing valid (throughput is data-shape-driven, not label-quality-driven) but is less externally "credible" than a published-dataset benchmark; the Higgs/Epsilon comparison stays deferred (see Deferred Ideas).

### How the numbers are produced (aggregate vs re-run)
- **D-03 (aggregate committed per-phase results + one new run that adds ONLY CatBoost-GPU):** Roll up the already-committed per-phase BENCH-02 `result.json` files (`bench/phase12_cuda_oracle/bench02-result.json`, `bench/phase13_cuda_oracle/result.json`, and any Phase 10/11 numbers if present) into the single BENCH-03 sign-off document. The **one new human-gated Kaggle CUDA run** adds **only** the missing **official CatBoost GPU** timing on the *same synthetic large-n configs* the per-phase runs used, so the head-to-head cell can be filled. Accepted consequence: the catboost-rs-device and host-CPU numbers come from prior sessions (Phase 12/13 P100 runs) while the CatBoost-GPU numbers come from this session — the doc must **label each number's source run/session** (hardware, date) so the mixed-session provenance is explicit, not hidden.

### Correctness gate scope (GPUT-14 / RESULTS.md backfill)
- **D-04 (assume green; speed-only; oracle as pre-flight only — do NOT own GPUT-14):** Per ROADMAP SC-2, the correctness oracle still runs as a **blocking pre-flight check** in this phase's notebook before any timing (a fast-but-wrong number is never quoted). BUT Phase 14 treats device correctness as **already established** by the Phase 12/13 real-P100 runs — it does **NOT** formally flip GPUT-14 to satisfied, and does **NOT** backfill the still-`TBD` depth-1/depth-6 oracle table in `bench/RESULTS.md`. That standing-gate/backfill debt is surfaced explicitly here as **out of scope for Phase 14** (flag for milestone-close audit / a follow-up, not this speed phase's job). Rationale: this phase is scoped to the BENCH-03 *speed* sign-off; conflating it with re-litigating the GPUT-14 correctness bookkeeping would widen scope past the roadmap requirement.

### Claude's Discretion
- **Official CatBoost GPU config-matching** — the exact parameter mapping between catboost-rs and the official `catboost` Python package (matching depth / iterations / learning rate / bootstrap / grow-policy / border_count so the two libraries run comparable configs) — research/planner resolves against the CatBoost docs + existing `benchmark.py` template. Note the two libraries' defaults differ; document any config that could NOT be matched.
- **BENCH-03 deliverable format & location** — whether the sign-off lands as an extension of `bench/RESULTS.md`, a new `bench/BENCH-03-SIGNOFF.md`, and/or a coverage/speed matrix table; and how the aggregated per-phase JSONs are stitched. Planner chooses; must include per-number source provenance (D-03) and the pre-Phase-10 host-light baseline reference.
- **Which per-phase families to include as matrix rows** beyond depth-6 RMSE/Logloss — bounded by which per-phase BENCH-02 numbers actually exist to aggregate.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements, roadmap & milestone framing
- `.planning/REQUIREMENTS.md` — **BENCH-03** (the sole requirement; comprehensive final speed-parity sign-off aggregating BENCH-02) + BENCH-01/BENCH-02 text + GPUT-14 status + the "validation authority = Kaggle CUDA" and "parity bar ε=1e-4" framing (lines ~11–14, 51–57).
- `.planning/ROADMAP.md` — Phase 14 Goal, Success Criteria 1–3, Notes (execution checklist: `nvidia-smi` → warm untimed fit → drain lazy queue → re-run oracle before timing; STACK.md/PITFALLS.md/`benchmark.py` template + Phase-10 `bench/` harness; `criterion 0.7.x` + optional `profiling` feature), standing landmines, Kaggle CUDA validation authority.
- `.planning/PROJECT.md` — v1.1 milestone goal (close the >20× GPU-training gap), the no-`cb-train`-dep-in-`cb-backend` landmine.
- `.planning/notes/gpu-training-host-light-root-cause.md` — the >20× **host-light** slowdown this milestone closes; the definition of the "pre-Phase-10 baseline" D-01 signs off against.

### Existing benchmark/harness assets (reuse; do NOT rebuild)
- `bench/RESULTS.md` — the human sign-off log (BENCH-01/02); D-10-09 escalation pin (depth-1 device≥CPU only at large n); the per-phase run-log format Phase 14 aggregates into. **Its depth-1/depth-6 oracle table is still all `TBD` — D-04 leaves that backfill out of scope.**
- `bench/cuda_oracle.ipynb` — the Phase-10 authoritative Kaggle CUDA oracle+speed notebook (the harness this phase extends for the CatBoost-GPU head-to-head).
- `bench/kaggle_cuda_phase13.ipynb`, `bench/phase13_cuda_oracle/{oracle.py,result.json,result.md}` — Phase-13 real-P100 sign-off (per-family correctness gate + BENCH-02 grow-loop speed) — a template for the notebook structure AND a source of aggregated numbers.
- `bench/phase12_cuda_oracle/{bench.py,oracle.py,bench02-result.json,correctness-result.json}` — Phase-12 real-P100 BENCH-02 speed (`bench_grow_speed_test.rs`, CB_BENCH=1, depthwise/region large-n 30–42×) — the aggregation source + the CatBoost-GPU-vs-config template.
- `bench/generator.py` — the large-n synthetic workload generator (D-02 matrix source).
- `bench/fixtures/` — depth-1/depth-6 expected trees + primitive oracles + `X_depth6_speed.npy` etc.
- `crates/cb-train/tests/bench_grow_speed_test.rs` — the device-vs-host-CPU grow-loop speed harness (CB_BENCH=1) invoked by the per-phase bench kernels; the anchor timing to aggregate.
- `benchmark.py`, `benchmark_fast.py`, `benchmark_small.py` (repo root) — the original benchmark templates referenced by the roadmap for the official-CatBoost comparison shape.
- `.planning/research/STACK.md`, `.planning/research/PITFALLS.md` — protocol/pitfall references named by the roadmap notes.

### Prior-phase context consumed as substrate
- `.planning/phases/13-.../13-CONTEXT.md` — the five device families feeding this aggregate; the **deferred "Higgs/Epsilon comprehensive aggregate" note that pointed here** (now scoped down to synthetic-only per D-02).
- `.planning/phases/12-.../12-CONTEXT.md`, `.planning/phases/11-.../11-CONTEXT.md`, `.planning/phases/10-.../10-CONTEXT.md` — per-phase BENCH-02 discipline, ε bars, landmines, `SPIKE-REDUCTION.md`.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Whole `bench/` harness is reusable as-is** — the Phase-10/12/13 Kaggle notebooks + `oracle.py`/`bench.py` kernels (repo→/tmp staging, `--no-default-features --features cuda`, `nvidia-smi`/`nvcc` env capture, compact `result.json` to `/kaggle/working`, correctness-gate-before-speed structure). Phase 14's new run is a *variant* of these (add a CatBoost-GPU timing arm), not a new harness.
- **`crates/cb-train/tests/bench_grow_speed_test.rs` (CB_BENCH=1)** — the existing device-vs-host-CPU grow-loop timing harness. The per-phase BENCH-02 numbers already come from it; Phase 14 aggregates its outputs and matches its synthetic configs when timing CatBoost GPU.
- **Committed per-phase `result.json` files** — machine-readable BENCH-02 numbers ready to aggregate (no re-run needed for the device/CPU columns per D-03).

### Established Patterns
- **Correctness-is-a-blocking-gate-before-speed** — every per-phase notebook halts on oracle failure before printing a speed number (T-10-25). D-04 keeps the oracle as a pre-flight check but not a GPUT-14 backfill.
- **Do-NOT-fabricate-numbers** — every recorded value comes from an actual run; `TBD` cells stay `TBD` until a real run fills them (`bench/RESULTS.md` header rule). D-03's mixed-session aggregation MUST label each number's source run.
- **Warm-run / JIT-excluded / train-only / queue-drained** timing protocol (roadmap execution checklist).

### Integration Points
- New work is **benchmark scripting + a sign-off document**, not production `cb-backend`/`cb-train` source changes. The `Runtime` seam, device kernels, and CPU path are all frozen (D-04 no-regression). The only code touched is under `bench/` (+ optionally a `profiling` dev feature / `criterion` dev-dep for in-env relative-timing regression, per roadmap).
- Official CatBoost GPU comparison uses the `catboost` Python package on the Kaggle CUDA instance (already how `benchmark.py` is shaped), run on the same synthetic large-n inputs.

</code_context>

<specifics>
## Specific Ideas

- One new human-gated Kaggle CUDA notebook = a variant of `bench/cuda_oracle.ipynb`/`kaggle_cuda_phase13.ipynb` that (1) runs the correctness oracle as a blocking pre-flight, then (2) times official CatBoost GPU on the same synthetic large-n configs the per-phase runs used.
- BENCH-03 sign-off doc aggregates `phase12_cuda_oracle/bench02-result.json` + `phase13_cuda_oracle/result.json` (+ any Phase 10/11) into one table: columns = device (cuda) s / host-CPU s / CatBoost-GPU s / speedup-vs-host-CPU / device≥20×?; each row labeled with its source run (hardware, date).
- Hard pass = every aggregated row shows device ≥20× vs host-CPU baseline; CatBoost-GPU column annotated informational.

</specifics>

<deferred>
## Deferred Ideas

- **Real named-dataset benchmark (Higgs / Epsilon) head-to-head vs CatBoost GPU** — considered (it was Phase-13's deferred pointer to here) and **declined for Phase 14** in favor of synthetic-large-n-only (D-02). Remains a credible post-milestone stretch deliverable if an externally-publishable benchmark is later wanted.
- **A hard "within X× of official CatBoost GPU" parity gate** — considered and declined (D-01: CatBoost-GPU is informational). Revisit only if the milestone owner later wants absolute-throughput parity as a gate.
- **Formally closing GPUT-14 + backfilling `bench/RESULTS.md`'s TBD depth-1/depth-6 oracle table** — explicitly OUT of scope for Phase 14 (D-04). This is standing correctness-bookkeeping debt to resolve at milestone-close audit or a dedicated follow-up, not in this speed-sign-off phase.
- **One fresh comprehensive re-run of the entire matrix in a single session** — considered and declined (D-03: aggregate + add-CatBoost-GPU-only) to save GPU time; the accepted cost is mixed-session provenance, mitigated by per-number source labels.

### Reviewed Todos (not folded)
- **Estimated-feature stored-border-VALUE quantization-grid parity** (`estimated-feature-grid-parity.md`) — NOT folded. It concerns the KNN estimated-feature online-HNSW port (FEAT-07, deferred to Phase 9), unrelated to Phase 14's speed benchmark. Out of scope.

</deferred>

---

*Phase: 14-comprehensive-kaggle-cuda-speed-benchmark-parity-sign-off*
*Context gathered: 2026-07-04*
