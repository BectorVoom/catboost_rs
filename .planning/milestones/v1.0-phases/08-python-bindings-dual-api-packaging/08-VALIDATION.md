---
phase: 8
slug: python-bindings-dual-api-packaging
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-06-21
---

# Phase 8 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x (Python surface) + cargo test (Rust binding crate) |
| **Config file** | none — Wave 0 installs (test venv: maturin, scikit-learn, numpy, pandas, pyarrow, polars; optional python3.13t) |
| **Quick run command** | `pytest -q` (after `maturin develop`) |
| **Full suite command** | `maturin develop && pytest && cargo test -p catboost-rs-py` |
| **Estimated runtime** | ~60–180 seconds (includes a maturin develop build) |

---

## Sampling Rate

- **After every task commit:** Run `pytest -q` (or `cargo test` for Rust-only tasks)
- **After every plan wave:** Run `maturin develop && pytest`
- **Before `/gsd-verify-work`:** Full suite must be green, incl. sklearn `check_estimator` and the python3.13t free-threaded phase gate (08-06) — or its recorded scoped deferral
- **Max feedback latency:** ~180 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | Status |
|---------|------|------|-------------|-----------|-------------------|--------|
| 8-01-01 | 01 | 0 | PYAPI-01 | infra | `maturin develop` builds the cpu wheel (Wave 0 venv-setup, Plan 08-01 Task 2 action) | ⬜ pending |
| 8-01-02 | 01 | 1 | PYAPI-01 | build | `cargo build -p catboost-rs-py --no-default-features --features cpu 2>&1 \| tail -5 && ! cargo tree -p catboost-rs-py --no-default-features --features rocm 2>&1 \| grep -q "cubecl-cpu\|/cpu"` | ⬜ pending |
| 8-01-03 | 01 | 1 | PYAPI-01, PYAPI-03, PYAPI-04 | py+rust | `cd crates/catboost-rs-py && ../../.venv-py8/bin/maturin develop --features cpu 2>&1 \| tail -3 && ../../.venv-py8/bin/pytest tests/test_smoke.py -q 2>&1 \| tail -15` | ⬜ pending |
| 8-02-01 | 02 | 2 | PYAPI-05 | py+rust | `cd crates/catboost-rs-py && ../../.venv-py8/bin/maturin develop --features cpu 2>&1 \| tail -2 && ../../.venv-py8/bin/pytest tests/test_errors.py -q 2>&1 \| tail -10 && cargo test -p catboost-rs-py errors 2>&1 \| tail -5` | ⬜ pending |
| 8-02-02 | 02 | 2 | PYAPI-03, PYAPI-05 | py+rust | `cd crates/catboost-rs-py && ../../.venv-py8/bin/maturin develop --features cpu 2>&1 \| tail -2 && ../../.venv-py8/bin/pytest tests/test_params.py -q 2>&1 \| tail -12 && cargo test -p catboost-rs-py params 2>&1 \| tail -5` | ⬜ pending |
| 8-03-01 | 03 | 3 | PYAPI-04, PYAPI-06 | py+rust | `cd crates/catboost-rs-py && ../../.venv-py8/bin/maturin develop --features cpu 2>&1 \| tail -2 && ../../.venv-py8/bin/pytest tests/test_ingestion.py -q 2>&1 \| tail -15 && cargo test -p catboost-rs-py ingest 2>&1 \| tail -5` | ⬜ pending |
| 8-03-02 | 03 | 3 | PYAPI-03 | py | `cd crates/catboost-rs-py && ../../.venv-py8/bin/maturin develop --features cpu 2>&1 \| tail -2 && ../../.venv-py8/bin/pytest tests/test_ingestion.py -q -k pool 2>&1 \| tail -10` | ⬜ pending |
| 8-04-01 | 04 | 4 | PYAPI-03 | py | `cd crates/catboost-rs-py && ../../.venv-py8/bin/maturin develop --features cpu 2>&1 \| tail -2 && ../../.venv-py8/bin/pytest tests/test_native_api.py -q 2>&1 \| tail -12` | ⬜ pending |
| 8-04-02 | 04 | 4 | PYAPI-03 | py (oracle ≤1e-5) | `cd crates/catboost-rs-py && ../../.venv-py8/bin/maturin develop --features cpu 2>&1 \| tail -2 && ../../.venv-py8/bin/pytest tests/test_oracle_parity.py -q 2>&1 \| tail -12` | ⬜ pending |
| 8-05-01 | 05 | 5 | PYAPI-02 | py+rust | `cd crates/catboost-rs-py && ../../.venv-py8/bin/maturin develop --features cpu 2>&1 \| tail -2 && ../../.venv-py8/bin/python -c "import catboost_rs,sklearn.base as b; e=catboost_rs.CatBoostRegressor(iterations=5); assert b.clone(e); ... ; print('ok')" 2>&1 \| tail -5 && cargo test -p catboost-rs-py estimator 2>&1 \| tail -5` | ⬜ pending |
| 8-05-02 | 05 | 5 | PYAPI-02 | py (check_estimator) | `cd crates/catboost-rs-py && ../../.venv-py8/bin/maturin develop --features cpu 2>&1 \| tail -2 && ../../.venv-py8/bin/pytest tests/test_check_estimator.py -q 2>&1 \| tail -20` | ⬜ pending |
| 8-06-01 | 06 | 6 | PYAPI-06 | py (skip-guard; 3.13t phase gate) | `cd crates/catboost-rs-py && ../../.venv-py8/bin/maturin develop --features cpu 2>&1 \| tail -2 && ../../.venv-py8/bin/pytest tests/test_free_threaded.py -q 2>&1 \| tail -8` | ⬜ pending |
| 8-06-02 | 06 | 6 | PYAPI-06 | doc-grep | `grep -v '^#' crates/catboost-rs-py/FREE_THREADING.md \| grep -c -e "own-before-detach" -e "gil_used" -e "free-threaded" -e "custom_loss"` | ⬜ pending |
| 8-07-01 | 07 | 6 | PYAPI-01 | build+doc-grep | `cd crates/catboost-rs-py && ../../.venv-py8/bin/maturin build --features cpu --release 2>&1 \| tail -5; ls ../../target/wheels/*.whl 2>/dev/null \| grep -c abi3; grep -v '^#' PACKAGING.md 2>/dev/null \| grep -c -e "two distribution" -e "catboost-rs-rocm" -e "mutually exclusive"` | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

**Manual-only task (excluded from the automated map):** 8-07-02 (build the rocm wheel in-env, `checkpoint:human-action`) — see Manual-Only Verifications. 8-06-01 additionally carries a `<human-check>` phase gate on `python3.13t` (concurrent buffer-safety) layered over its automated skip-guard.

**Sampling continuity:** every `auto` task across all six waves carries an `<automated>` verify — there is no run of 3 consecutive tasks without an automated check (the only non-automated task, 8-07-02, is a terminal human-action checkpoint, and its sibling 8-07-01 in the same wave is automated). Continuity holds.

---

## Wave 0 Requirements

- [x] Covered by Plan 08-01 Task 2 action (venv-setup): `python3 -m venv .venv-py8 && .venv-py8/bin/pip install "maturin>=1.9.4,<2.0" scikit-learn numpy pandas pyarrow polars`
- [x] `tests/conftest.py` — shared fixtures (toy datasets, oracle vectors ≤1e-5) created across 08-01/08-04
- [x] `pytest` framework install + `pyproject.toml`/`maturin` project scaffold (08-01 Task 2)
- [ ] Optional `python3.13t` (free-threaded interpreter) for the PYAPI-06 buffer-safety phase gate — built/installed in-env if available, else scoped deferral (08-06)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| rocm per-backend wheel build | PYAPI-01 | Requires the in-env HIP/ROCm toolchain (gfx1100); not in GitHub CI | Build `catboost-rs-rocm` wheel in-env (`maturin build --no-default-features --features rocm --release`), import `catboost_rs`, run a smoke fit/predict (08-07 Task 2) |
| free-threaded concurrent fit/predict | PYAPI-06 | Standard test venv is a GIL build; needs `python3.13t` built/installed in-env | `python3.13t -m pytest tests/test_free_threaded.py` runs (not skipped) and passes; if 3.13t unavailable, record scoped deferral in 08-06-SUMMARY (PYAPI-06 stays code-property-validated) |

*Per-requirement validation detail lives in 08-RESEARCH.md `## Validation Architecture`.*

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (test venv via 08-01 Task 2)
- [x] No watch-mode flags
- [x] Feedback latency < 180s
- [x] `nyquist_compliant: true` set in frontmatter
- [x] `wave_0_complete: true` set in frontmatter (Wave 0 = 08-01 venv-setup action)

**Approval:** approved
