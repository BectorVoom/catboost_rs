# Packaging `catboost_rs` (PYAPI-01)

This document is the realization of the **two distribution** packaging model
(D-08 / D-09) for the `catboost_rs` Python extension: how the cpu and ROCm
backends ship as wheels, why there are two PyPI distributions, the
abi3-cpu-primary deliverable, and the build/publish split (cpu/abi3 wheel in CI,
ROCm wheel in-env only).

## The two-distribution model (D-08)

PyPI accepts exactly **one** binary wheel per `name + version + platform tag`.
A Python *extra* pulls additional *Python* dependencies — it cannot select an
alternate compiled `.so`. The cpu and ROCm backends are two *different* compiled
artifacts (cpu links nothing GPU; ROCm links HIP/`cubecl-hip`), so they cannot
coexist under one distribution name. They therefore ship as **two distributions**:

| Distribution      | Backend | Wheel built where        | `pyproject` file       | Import name   |
| ----------------- | ------- | ------------------------ | ---------------------- | ------------- |
| `catboost-rs`     | cpu     | CI (GitHub Actions)      | `pyproject.toml`       | `catboost_rs` |
| `catboost-rs-rocm`| ROCm    | in-env (gfx1100/ROCm 7.1)| `pyproject-rocm.toml`  | `catboost_rs` |

Both distributions expose the **same** import name `catboost_rs` (D-09 — never
shadow the real upstream `catboost` package). The `catboost-rs[rocm]` extra is a
dependency edge:

```toml
# pyproject.toml
[project.optional-dependencies]
rocm = ["catboost-rs-rocm"]
```

so `pip install "catboost-rs[rocm]"` pulls the `catboost-rs-rocm` distribution
which carries the HIP-linked `.so`.

## Mutually exclusive — never install both (Pitfall 4)

Because `catboost-rs` (cpu) and `catboost-rs-rocm` (ROCm) **both** install a
module named `catboost_rs`, installing both into the same environment leaves the
import resolution of `catboost_rs` undefined — whichever `.so` the installer
wrote last wins, silently. **The two distributions are mutually exclusive and
MUST NOT be installed simultaneously in the same interpreter / virtualenv.**

- For cpu: `pip install catboost-rs`
- For ROCm: `pip install catboost-rs-rocm` (equivalently `pip install "catboost-rs[rocm]"`)
- Never both.

This is the documented mitigation for STRIDE threat T-08-22 (import-name
confusion): the extra pulls a *separate distribution*, the two are never
co-installed, and this warning is the user-facing contract.

## abi3-py312 cpu wheel is the primary deliverable

The cpu distribution is built as an **abi3 (limited-API) wheel** pinned at the
`abi3-py312` floor (the `pyo3` `abi3-py312` Cargo feature, set in
`Cargo.toml`). One abi3-py312 wheel is forward-compatible across **CPython
3.12 / 3.13 / 3.14 GIL builds** — a single artifact instead of one wheel per
minor version. This is the PYAPI-01 primary deliverable.

```bash
maturin build --features cpu --release
# -> target/wheels/catboost_rs-<ver>-cp312-abi3-<platform>.whl
```

The wheel tag `cp312-abi3-<platform>` is what proves abi3: `cp312` is the floor,
`abi3` is the limited-API marker, and pip will install it on any CPython >= 3.12
(GIL).

### Free-threaded wheel is DEFERRED (abi3 ⊥ free-threading)

abi3 and the free-threaded (no-GIL) ABI are **mutually exclusive** in PyO3 0.29:
the free-threaded build uses a new ABI with no limited-API equivalent, so an
abi3 wheel cannot load on a `python3.13t`/`python3.14t` interpreter (Pitfall 1).

Per CONTEXT (Deferred Ideas) and the A2 decision, the free-threaded *wheel* is
**deferred**. PYAPI-06 (free-threaded-aware design) is satisfied as a **code
property** — own-before-detach ingest (D-11) plus `#[pymodule(gil_used = false)]`
— not as a shipped free-threaded wheel. See [FREE_THREADING.md](FREE_THREADING.md)
for the full rationale and the one documented caveat (the custom-loss callback
path).

## Build / publish split

| Wheel        | Built by                         | Why                                                                 |
| ------------ | -------------------------------- | ------------------------------------------------------------------- |
| cpu / abi3   | CI — `.github/workflows/python-wheels.yml` | GitHub-hosted runners have Python + Rust; no GPU toolchain needed.  |
| ROCm         | **in-env only** (gfx1100/ROCm 7.1) | The ROCm wheel links HIP (`cubecl-hip-sys`) and needs the ROCm toolchain GitHub runners do not have. |

The ROCm wheel is **NEVER** built in GitHub Actions (D-06, carried from Phase 7:
the `rocm` feature requires the HIP/ROCm toolchain + an AMD GPU and is the
authoritative GPU gate validated locally/manually in-env on gfx1100 only). CI
builds the cpu/abi3 wheel exclusively.

### Building the cpu wheel (CI and locally)

```bash
cd crates/catboost-rs-py
maturin build --features cpu --release
```

### Building the ROCm wheel (in-env only)

The ROCm distribution uses `pyproject-rocm.toml` (`name = "catboost-rs-rocm"`,
`module-name = "catboost_rs"`). The backend feature wiring from 08-01 makes
`--no-default-features --features rocm` cpu-free (no `cubecl-cpu` in the tree):

```bash
cd crates/catboost-rs-py
# Swap the rocm pyproject in for the build (it names the distribution catboost-rs-rocm):
maturin build --manifest-path Cargo.toml --no-default-features --features rocm --release
# Smoke-test in a fresh venv on the GPU runtime:
#   pip install target/wheels/catboost_rs_rocm-*.whl
#   python -c "import catboost_rs; m = catboost_rs.CatBoostRegressor(iterations=5); ..."
```

Compile/link only needs the HIP toolchain at build time; a live GPU is needed
for the import/fit/predict smoke at runtime.

## Threat-model alignment

- **T-08-21** (cpu leaks into the ROCm wheel): mitigated by the 08-01 feature
  forwarding — `--no-default-features --features rocm` is cpu-free
  (`cargo tree` gate + the in-env build).
- **T-08-22** (both wheels co-installed → undefined `.so` resolution): mitigated
  by this document's mutual-exclusivity contract and the separate-distribution
  `[rocm]` extra.
- **T-08-23** (ROCm build in untrusted CI exposes the HIP toolchain): accepted —
  the ROCm wheel is built in-env only, never in GitHub Actions (D-06).
