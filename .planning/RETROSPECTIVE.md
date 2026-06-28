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

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.0 | many | 8 | Established per-stage oracle gating + instrumented-trainer root-causing; split large phases (6, 7) into sub-phases |

### Cumulative Quality

| Milestone | Tests | Coverage | Zero-Dep Additions |
|-----------|-------|----------|-------------------|
| v1.0 | hundreds (cb-train/cb-compute/cb-core suites + Rust+pytest oracle gates) | parity-gated ≤1e-5 (CPU), ε=1e-4 (GPU) | in-house pairwise Cholesky, reduction utility, TFastRng64 port (no external ANN/RNG deps) |

### Top Lessons (Verified Across Milestones)

1. Specify parity as correctness **and** performance — numerical gates alone let a slow-but-correct implementation pass.
