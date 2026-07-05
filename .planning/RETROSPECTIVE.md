# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.0 — Core Parity

**Shipped:** 2026-06-28
**Phases:** 8 (1–8, with Phase 6 & 7 split into 6.1–6.6 / 7.1–7.6 sub-phases) | **Plans:** ~57 | **Sessions:** many (multi-week)

### What Was Built
- A Rust-native CatBoost achieving oracle-locked (≤1e-5) CPU parity across the full training core: plain + ordered boosting, oblivious + non-symmetric trees, four leaf-estimation methods, bootstrap/sampling, regularization, overfitting detection.
- The complete loss/metric/feature matrix — regression, binary, multiclass/multilabel, six ranking losses, text/embedding features, CTR/categoricals, SHAP, score functions, uncertainty.
- GPU **structural** parity via CubeCL (cuda/rocm/wgpu, generic runtime), rocm-validated to ε=1e-4 vs the CPU path.
- A dual-surface (sklearn + CatBoost-native) PyO3/maturin Python binding with multi-source ingest and per-backend wheels.

### What Worked
- **Per-stage oracle gating** (quantization borders → splits → leaf values → staged approx → predictions) caught divergences at the earliest stage rather than only at final predictions — the single highest-leverage practice for parity work.
- **Instrumented upstream trainer** (rebuilt catboost 1.2.10 `_catboost.so` with `CB_INSTRUMENT_LOG`) turned "close but not bit-exact" mysteries into root-caused fixes (pc=4 partition, StochasticRank per-tree noise, KNN online-HNSW discovery).
- **Escalate-don't-weaken** discipline: no `#[ignore]`, no relaxed tolerances to make gates pass; gaps were deferred honestly with documented root cause.

### What Was Inefficient
- The live ROADMAP was trimmed to current phases as work progressed, so the milestone-close CLI undercounted v1.0's true scope (saw 2 phases, not 8). Future: keep a milestone-grouped ROADMAP rather than trimming.
- The GPU phase (7) shipped a *derivatives-only* MVP whose host-light grow loop was never wired for performance — a structural-parity success that left a >20× speed gap discovered only later. The "structural vs performance parity" distinction should have been an explicit success criterion split.
- Repeated "gsd-tools CLI absent → updated STATE/ROADMAP manually" notes indicate tooling availability churn across sessions.

### Patterns Established
- CubeCL kernels are written once and compiled to cuda/rocm/wgpu; **correctness validated in-env on AMD/ROCm, performance benchmarked on CUDA elsewhere** (no NVIDIA in-env).
- **Landmine:** never add a `cb-train` dependency to `cb-backend` — feature unification breaks the rocm runtime; transcribe CPU references inline.
- CubeCL HIP: no `-inf` literal in `#[cube]` kernels (gfx1100 JIT rejects); use a finite sentinel.

### Key Lessons
1. **"Parity" must specify correctness *and* performance up front.** Structural/numerical parity (≤1e-5) was rigorously gated; throughput was not a success criterion, so an MVP that idles the GPU passed — and is now its own milestone.
2. **Per-stage oracles + an instrumented reference trainer** are the combination that makes bit-exact reverse-engineering tractable.
3. **Defer honestly.** Carrying FEAT-07 (HNSW) and GPU-perf as documented gaps kept the parity bar credible rather than papering over it.

### Cost Observations
- Model mix: predominantly opus (quality profile) for planning/execution; sonnet for synthesis.
- Notable: instrumented-trainer rebuilds were expensive but repeatedly decisive — worth the cost on parity blockers.

---

## Milestone: v1.1 — GPU Performance

**Shipped:** 2026-07-05
**Phases:** 5 (10–14) | **Plans:** 36 | **Tasks:** 64

### What Was Built
- The CatBoost boosting inner loop moved from a derivatives-only host-light MVP onto a fully device-resident CubeCL training path, reversing the pre-Phase-10 >20× device-slower-than-CPU gap into a **23.9×–42.1× speedup vs the host-light CPU baseline** on Tesla P100 (BENCH-03: PASS).
- A from-scratch CubeCL device-primitive library (no CUB) — scan/segmented-scan, reduce/reduce-by-key, radix sort + stable 1-bit reorder, bit-compression, partition-update, stat aggregation — with a deterministic fixed-point-u64 reduction.
- A bit-packed device-resident compressed index + a per-fit `GpuTrainSession` keeping the quantized matrix, gradients, and approx resident across iterations (no per-tree re-upload or `der1` read-back; only O(1) BestSplit + 2^depth part-stats cross per level).
- Full device coverage behind an `Ok(None)`→CPU per-fit fallback: depth-1→depth-6 partition-aware histograms + subtraction trick + Newton der2/Exact leaves; Depthwise/Lossguide/Region grow policies; bootstrap/random-strength/MVS sampling; CTR/categoricals; PairLogit + batched device Cholesky; query/listwise ranking; multiclass/multi-target/uncertainty; ordered boosting; Langevin/SGLB noise.

### What Worked
- **The `Ok(None)`→CPU per-fit fallback gate** let each device family land independently and stay oracle-safe — any uncovered config transparently falls back to the CPU reference, so partial device coverage never risks a wrong model.
- **Device-residency-first design** (upload once, recompute `der1` on device, cross only O(1) descriptors per level) is what actually closed the >20× gap — the derivatives-only MVP's host↔device round-trips were the root cause.
- **Per-family self-oracles in-env on real gfx1100** (≤1e-4, many bit-exact 0.0 divergence) caught divergences immediately, before the human-gated Kaggle CUDA runs — the ROCm smoke path paid for itself as a fast local gate even though it is not the authoritative oracle.
- **Orchestrator-driven Kaggle CUDA runs** (the `kaggle` CLI pipeline proven in Phase 12, reused through 14) turned the human-gated external oracle into a repeatable step rather than a per-phase blocker.

### What Was Inefficient
- **The standing per-phase speed check (BENCH-02) was not enforced uniformly** — Phase-10 (depth-1) and Phase-11 (depth-6) BENCH-02 Kaggle rows were never run, so the milestone closed with GPUT-14's aggregate sign-off Pending and the BENCH-03 aggregate stitching only the committed Phase-12/13 numbers. A "standing gate" mapped to one phase but enforced onward is easy to let slip on the early phases; it needs an explicit per-phase checklist item, not just a milestone-level requirement.
- **ROADMAP live-trimming recurred** — Phase 11 showed `4/5` / "In Progress" in the live Progress table at close despite being complete, a repeat of the v1.0 undercount; the milestone-close CLI and archive had to be corrected by hand.
- **Root filesystem at 100%** blocked `git commit` during Phase 13 (Rule-3 blocker) — disk pressure is a recurring in-env hazard for this repo (the vendored `catboost-master/` + large target/ dirs).

### Patterns Established
- **Never add a `cb-train` dependency to `cb-backend`** — Cargo feature unification breaks the rocm runtime; transcribe CPU references inline. The `Runtime` seam stays CubeCL-free (plain host structs cross the boundary).
- **No `-inf` literals inside `#[cube]` kernels** — HIP gfx1100 JIT rejects them (CUDA accepts, so it's invisible to cpu/wgpu cargo check); use a finite `f32::MIN` sentinel.
- **A deterministic reduction is required even where f64 atomic-add exists** — CUDA has it but atomicAdd commit ordering is still non-deterministic and compounds over hundreds of trees; the fixed-point-u64 accumulator holds ε=1e-4 parity.
- **Serial `#[cube]` transcription of CPU RNG streams** (TFastRng64 / PCG) reproduces bootstrap/MVS/YetiRank draws bit-for-bit on device — kept device-resident with no per-tree host mask round-trip.

### Key Lessons
1. **Residency, not kernels, was the performance lever.** The structural-parity kernels already existed after v1.0 Phase 7; the >20× win came from keeping data on-device across the whole fit and eliminating the per-tree `der1` read-back — architecture over micro-optimization.
2. **A "standing gate" needs a per-phase artifact, not just a milestone requirement.** GPUT-14 / BENCH-02 were defined as enforced-every-phase but tracked at the milestone level, so the early-phase rows silently went un-run and became accepted close-time debt.
3. **A fast non-authoritative local gate (ROCm) accelerates an expensive authoritative one (Kaggle CUDA).** Correctness was de-risked in-env before every human-gated CUDA run, so the CUDA runs mostly confirmed rather than discovered.

### Cost Observations
- Model mix: predominantly opus (quality profile) for planning/execution; sonnet for synthesis.
- Sessions: multi-day (2026-06-28 → 2026-07-05), 203 commits across the milestone range.
- Notable: the orchestrator drove the Kaggle CUDA notebook runs itself (git-archive tracked-source-only tarball → small dataset, background-poll kernel status) — the human gate became a repeatable CLI step rather than a hand-off.

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.0 | many | 8 | Established per-stage oracle gating + instrumented-trainer root-causing; split large phases (6, 7) into sub-phases |
| v1.1 | multi-day (203 commits) | 5 (10–14) | Device-residency-first GPU architecture + `Ok(None)`→CPU per-fit fallback gate; orchestrator-driven Kaggle CUDA runs via the `kaggle` CLI |

### Cumulative Quality

| Milestone | Tests | Coverage | Zero-Dep Additions |
|-----------|-------|----------|-------------------|
| v1.0 | hundreds (cb-train/cb-compute/cb-core suites + Rust+pytest oracle gates) | parity-gated ≤1e-5 (CPU), ε=1e-4 (GPU) | in-house pairwise Cholesky, reduction utility, TFastRng64 port (no external ANN/RNG deps) |
| v1.1 | + device self-oracle suites (44 device tests ALL-PASS on P100; per-family in-env gfx1100 ≤1e-4) | device path ε=1e-4 vs CPU (depth-1 ≤1e-5); CPU byte-unchanged (D-04) | from-scratch CubeCL device-primitive library (no CUB), batched device Cholesky, fixed-point-u64 deterministic reduction, `#[cube]` PCG/TFastRng64 transcriptions |

### Top Lessons (Verified Across Milestones)

1. Specify parity as correctness **and** performance — numerical gates alone let a slow-but-correct implementation pass. _(v1.0 lesson; v1.1 then closed the perf gap — via residency architecture, not new kernels.)_
2. A "standing gate" enforced every phase needs a **per-phase artifact**, not just a milestone-level requirement — else the early-phase rows silently go un-run (v1.1: GPUT-14 aggregate + Phase-10/11 BENCH-02 became close-time debt).
3. Keep a **milestone-grouped ROADMAP** rather than live-trimming to current phases — the undercount/stale-status bug recurred in both v1.0 and v1.1 closes.
