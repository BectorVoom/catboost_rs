# Milestones

## v1.1 GPU Performance (Shipped: 2026-07-05)

**Scope:** Phases 10–14, 36 plans, 64 tasks, 25 requirements (GPUT-01..22 + BENCH-01..03). Re-scoped in place 2026-07-02 against `CATBOOST_CUDA_KERNELS_DESIGN.md` (17 → 25). Full per-phase detail: `milestones/v1.1-ROADMAP.md`; per-requirement record: `milestones/v1.1-REQUIREMENTS.md`.

**Delivered:** The CatBoost boosting inner loop moved from a derivatives-only host-light MVP onto a fully device-resident CubeCL training path, reversing the pre-Phase-10 >20× device-slower-than-CPU gap into a **23.9×–42.1× speedup vs the host-light CPU baseline** on Tesla P100 (BENCH-03: PASS). Built from scratch (no CUB): a portable CubeCL device-primitive library (scan / segmented-scan, reduce / reduce-by-key, radix sort + stable 1-bit reorder, bit-compression, partition-update, stat aggregation) with a deterministic reduction; a bit-packed device-resident compressed index; a per-fit `GpuTrainSession` keeping the quantized matrix, gradients, and approx device-resident across iterations (no per-tree re-upload or `der1` read-back); depth-1 → depth-6 partition-aware histograms with the subtraction trick and Newton der2; and full device coverage of grow policies (Depthwise/Lossguide/Region), Exact weighted-quantile leaves, bootstrap/random-strength/MVS sampling, CTR/categoricals, PairLogit + batched device Cholesky, query/listwise ranking, multiclass/multi-target/uncertainty, ordered boosting, and Langevin/SGLB noise — each behind an `Ok(None)`→CPU per-fit fallback gate. The whole surface was self-oracled ε=1e-4 in-env on AMD gfx1100 and signed off on real Kaggle CUDA (P100) runs; the CPU path stays byte-unchanged (D-04).

**Key accomplishments (by phase):**

- **Phase 10 — Foundations:** From-scratch CubeCL device-primitive substrate + reduction-determinism spike (fixed-point u64 winner), the bit-packed `WriteCompressedIndex` cindex (GPUT-15), and a depth-1 oblivious tree grown fully on device over a residency-holding `GpuTrainSession` reachable from the public `fit()` — plus the reproducible Kaggle CUDA correctness+speed harness (BENCH-01/02) every later phase reuses.
- **Phase 11 — Depth>1 keystone:** `fullPass=false` leaf-keyed histograms + the subtraction trick + the LOCKED fixed-point `Atomic<u64>` accumulator (GPUT-06) grow a full depth-6 RMSE tree bit-exact and a Logloss tree via device Newton der2 (GPUT-07), ≤1e-4 on real gfx1100.
- **Phase 12 — Grow-policy/leaf/sampling/CTR:** Depthwise, Lossguide, and Region non-symmetric device grows (GPUT-18), device Exact weighted-quantile leaves (GPUT-19), on-device bootstrap/random-strength + MVS (CatBoost's default sampler, GPUT-17/09), and device CTR accumulation (GPUT-10) — each flipped from `Ok(None)` to a real device path, bit-exact/≤1e-4.
- **Phase 13 — Loss-family/multi-output/ordered:** PairLogit with a batched f64 device Cholesky solver (GPUT-11/21), five query/listwise ranking objectives incl. stochastic YetiRank/PFound-F over device query-grouping (GPUT-22), multiclass/multi-target/uncertainty K-dim Newton block-leaves (GPUT-12), ordered boosting (GPUT-13), and Langevin/SGLB noise (GPUT-20).
- **Phase 14 — Sign-off:** BENCH-03 signed off PASS — all 12 aggregated device rows 23.9×–42.1× vs the host-light CPU baseline on P100, CUDA correctness (44 device self-oracle tests, ALL-PASS) gated before any speed number, with an informational CatBoost-GPU head-to-head and full mixed-session provenance.

### Known Gaps (proceeded with incomplete requirements — formal accept-as-delivered)

Milestone closed with a documented override (recorded in `.planning/milestones/v1.1-phases/14-.../14-VERIFICATION.md`; standing debt in `.planning/PROJECT.md`). The device coverage passes per-family self-oracles ≤1e-4 in-env and on the committed Kaggle CUDA (P100) runs, but two milestone-wide sign-off rows were not executed:

- **GPUT-14** (ε=1e-4 device-vs-CPU standing correctness gate, Phase 11 onward) — status **Pending**: the milestone-wide Kaggle CUDA GPUT-14 row was never run as a single aggregate; coverage is evidenced per-family instead.
- **Phase-10 (depth-1) + Phase-11 (depth-6) BENCH-02** Kaggle CUDA speed runs were never executed; the BENCH-03 aggregate stitches the committed Phase-12/13 numbers only.

**Known deferred items at close:** 8 (see STATE.md → Deferred Items) — 1 pending requirement (GPUT-14), 2 un-run BENCH-02 rows, 1 human-needed verification + 1 UAT gap (Phase-10 depth-1 Kaggle gate), 2 quick tasks (1 resolved-stale, 1 superseded by FEAT-07), 1 pending todo (FEAT-07 HNSW, Phase 9).

---

## v1.0 Core Parity (Shipped: 2026-06-28)

**Scope:** Phases 1–8 (the full CPU parity surface). The archived `milestones/v1.0-ROADMAP.md` reflects only Phases 8–9 because earlier phases were trimmed from the live ROADMAP as they completed; the authoritative per-requirement record is `milestones/v1.0-REQUIREMENTS.md` (61/62 v1 requirements complete).

**Delivered:** A Rust-native CatBoost with oracle-locked (≤1e-5) parity across the CPU training core (plain + ordered boosting, oblivious + non-symmetric trees, four leaf-estimation methods, bootstrap/sampling, regularization, overfitting detection), the full loss/metric/feature matrix (regression, binary, multiclass/multilabel, six ranking losses, text/embedding features, CTR/categoricals, SHAP, score functions, uncertainty), model save/load, a Rust Builder API, GPU **structural** parity via CubeCL (cuda/rocm/wgpu, rocm-validated, ε=1e-4 vs CPU), and a dual-surface (sklearn + CatBoost-native) PyO3/maturin Python binding with per-backend wheels.

### Known Gaps (carried forward)

- **FEAT-07** — KNN estimated-feature bit-exact parity requires an online-HNSW port (~832 LOC C++); shipped with a brute-force-exact calcer that diverges from upstream's approximate HNSW. Carried as **Phase 9** planning context (deferred backlog, to be re-surfaced as its own milestone).
- **GPU performance parity** — GPU shipped as a *derivatives-only* MVP; the tree-growth inner loop still runs on the host CPU (>20× slower than official CatBoost GPU). This is **new scope** addressed by the next milestone, not a v1.0 regression. See `.planning/notes/gpu-training-host-light-root-cause.md`.

**Known deferred items at close:** 11 (see STATE.md → Deferred Items) — 4 UAT gaps, 4 human-needed verification sign-offs (incl. Phase 8 free-threaded run needing python3.13t), 2 quick tasks, 1 pending todo (the FEAT-07 HNSW work).

**Key accomplishments (Phase 8 — Python bindings, the final v1.0 phase):**

- A real `CatBoostRegressor().fit(X32, y32).predict(X32)` travels the entire NumPy -> OwnedColumns -> CatBoostBuilder::fit -> Model::predict -> NumPy boundary through the live catboost-rs facade, packaged as a maturin abi3-py312 cdylib that builds cpu-free under `--features rocm`.
- The binding is now honest about what it supports: every facade `CatBoostError` variant maps to a specific catchable Python exception (PYAPI-05), and the full 119-param upstream vocabulary is validated at `fit()` (D-06) so a known-but-unimplemented param (`nan_mode`) is rejected as a parity gap, a typo (`iteratons`) suggests `iterations`, and sklearn aliases (`n_estimators`/`max_depth`/`reg_lambda`) resolve.
- A user can now fit/predict from a NumPy array, a Pandas DataFrame, a pyarrow Table, or a Polars DataFrame — all converging on the existing `OwnedColumns::into_pool()` seam with equal predictions for equal data — while float64 / non-contiguous / ambiguous-object / nullable inputs are rejected with an actionable `CatBoostValueError`, every buffer is copied into owned Rust memory before any GIL release (PYAPI-06 as a code property), and a native `Pool(data, label, cat_features=...)` mirrors upstream `Pool.__init__`.
- A user can now fit a `CatBoostClassifier` (defaulting to Logloss) and get `(n,)` class labels + `(n,2)` probabilities, fit a `CatBoostRanker` on a `group_id` `Pool` and get `(n,)` ranking scores (with a group-less dataset rejected by an actionable `CatBoostValueError`), and the whole Python surface is parity-locked: `CatBoostRegressor`/`CatBoostClassifier.load_model(path)` load the offline catboost 1.2.10 reference `.cbm`/`.json` and reproduce its predictions to within 1e-5 (observed bit-exact) — hermetically, with no live `catboost` import and no re-fit fallback.
- The CatBoost-native estimators are now drop-in scikit-learn estimators: `get_params`/`set_params` round-trip the verbatim kwargs exactly (so `sklearn.base.clone`, `Pipeline`, and `GridSearchCV` work), `__sklearn_tags__` returns the sklearn >=1.6 `Tags` dataclass with the right `estimator_type`, predict-before-fit raises a `NotFittedError` (a `ValueError`), and sklearn's authoritative `check_estimator` passes every STRUCTURAL check while the dtype/contiguity/sparse checks are an explicit, enumerated, per-check-justified `xfail` allowlist (D-04) — not a blanket skip, so any NEW non-allowlisted contract regression fails the gate.
- `#[pymodule(gil_used = false)]` declared and backed by the 08-03 own-before-detach discipline, with a concurrent fit/predict buffer-safety test (GIL-build skip-guarded) and a FREE_THREADING.md documenting the abi3-vs-free-threaded wheel deferral and the custom-loss GIL-reentry caveat.
- The abi3-py312 cpu wheel (`catboost_rs-0.1.0-cp312-abi3-manylinux_2_39_x86_64.whl`) builds, installs into a fresh venv, and `import catboost_rs` exposes `CatBoostRegressor`; the two-distribution layout (cpu `catboost-rs` + rocm `catboost-rs-rocm`, both importing `catboost_rs`, mutually exclusive) is documented, CI builds the cpu wheel only, and the rocm distribution CONFIG (`pyproject-rocm.toml`) is shipped — the rocm wheel BUILD is deferred to gap plan 08-08 (generic GPU backend) because the facade `fit()` train path has no GPU `Runtime` implementation.

---
