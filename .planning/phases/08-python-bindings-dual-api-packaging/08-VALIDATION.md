---
phase: 8
slug: python-bindings-dual-api-packaging
status: draft
nyquist_compliant: false
wave_0_complete: false
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
| **Full suite command** | `maturin develop && pytest && cargo test -p <pyo3-crate>` |
| **Estimated runtime** | ~60–180 seconds (includes a maturin develop build) |

---

## Sampling Rate

- **After every task commit:** Run `pytest -q` (or `cargo test` for Rust-only tasks)
- **After every plan wave:** Run `maturin develop && pytest`
- **Before `/gsd-verify-work`:** Full suite must be green, incl. sklearn `check_estimator`
- **Max feedback latency:** ~180 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 8-01-01 | 01 | 0 | PYAPI-01 | — | N/A | infra | `maturin develop` builds the cpu wheel | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

*Planner expands this map per task — every PYAPI-01..06 requirement maps to at least one automated check below.*

---

## Wave 0 Requirements

- [ ] Test venv with `maturin`, `scikit-learn`, `numpy`, `pandas`, `pyarrow`, `polars` (none currently installed)
- [ ] Optional `python3.13t` (free-threaded interpreter) for the PYAPI-06 buffer-safety test
- [ ] `tests/conftest.py` — shared fixtures (toy datasets, oracle vectors ≤1e-5)
- [ ] `pytest` framework install + `pyproject.toml`/`maturin` project scaffold

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| rocm per-backend wheel build | PYAPI-01 | Requires the in-env HIP/ROCm toolchain (gfx1100); not in GitHub CI | Build `catboost-rs-rocm` wheel in-env, import `catboost_rs`, run a smoke predict |

*Per-requirement validation detail lives in 08-RESEARCH.md `## Validation Architecture`.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 180s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
