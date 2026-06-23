### Phase 8: Python Bindings, Dual API & Packaging

**Goal**: Python ML practitioners can drop catboost-rs into existing scikit-learn or CatBoost workflows via a dual-surface PyO3 binding distributed as per-backend wheels.
**Mode:** mvp
**Depends on**: Phase 7
**Requirements**: PYAPI-01, PYAPI-02, PYAPI-03, PYAPI-04, PYAPI-05, PYAPI-06
**Plans:** 7/7 plans complete
Plans:
**Wave 1**

- [x] 08-01-PLAN.md — Walking skeleton: scaffold the PyO3 crate + facade feature passthrough + test venv + thinnest CatBoostRegressor.fit(numpy).predict() end-to-end (PYAPI-01/03/04) — COMPLETE (commits 1526805 / 9b16c4c; Rust 4/4 + pytest 5/5 green; cpu-free rocm verified)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 08-02-PLAN.md — Typed exception taxonomy + full param-vocabulary registry validating at fit() (PYAPI-05, PYAPI-03; D-05/D-06/D-07)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 08-03-PLAN.md — Multi-source ingestion (NumPy/Pandas/Arrow/Polars) + strict validation + own-before-detach + native Pool (PYAPI-04/06/03; D-10/D-11/D-12)

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 08-04-PLAN.md — CatBoostClassifier + CatBoostRanker + Python-surface oracle parity ≤1e-5 (PYAPI-03; D-01)

**Wave 5** *(blocked on Wave 4 completion)*

- [x] 08-05-PLAN.md — sklearn contract (get/set_params, __sklearn_tags__, clone, NotFitted) + check_estimator gate with documented-skip allowlist (PYAPI-02; D-03/D-04)

**Wave 6** *(blocked on Wave 5 completion)*

- [x] 08-06-PLAN.md — Free-threaded-aware design: gil_used=false + multi-thread buffer-safety test (3.13t) + caveat docs (PYAPI-06) — COMPLETE (commits 733546f Task1 / fedf1b3 Task2; #[pymodule(gil_used = false)] backed by 08-03 own-before-detach; test SKIPs cleanly on the GIL venv; FREE_THREADING.md documents abi3⊥free-threaded wheel deferral + custom_loss GIL-reentry caveat. SCOPED DEFERRAL: no python3.13t in-env -> concurrent free-threaded RUN deferred; PYAPI-06 code-property-validated. Gates: maturin abi3-py312 OK; pytest 73 passed/5 skipped/79 xfailed; cargo 29/29)
- [x] 08-07-PLAN.md — Packaging: abi3-py312 cpu wheel + in-env rocm wheel under the two-distribution layout (PYAPI-01; D-08/D-09)
