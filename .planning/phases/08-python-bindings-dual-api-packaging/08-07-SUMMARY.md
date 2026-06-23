---
phase: 08-python-bindings-dual-api-packaging
plan: 07
subsystem: packaging
tags: [maturin, packaging, wheels, abi3, rocm, ci]

# Dependency graph
requires:
  - phase: 08-01
    provides: "catboost-rs-py maturin abi3-py312 cdylib + backend feature passthrough (cpu-free rocm dep tree)"
  - phase: 08-06
    provides: "gil_used=false module + FREE_THREADING.md (abi3-vs-free-threaded deferral)"
provides:
  - "crates/catboost-rs-py/pyproject.toml finalized cpu distribution (catboost-rs, abi3-py312, [rocm] extra)"
  - "crates/catboost-rs-py/pyproject-rocm.toml rocm distribution config (catboost-rs-rocm, module-name=catboost_rs)"
  - "crates/catboost-rs-py/PACKAGING.md two-distribution model + mutual exclusivity + build/publish split"
  - ".github/workflows/python-wheels.yml cpu/abi3 wheel build + import smoke (no rocm in Actions)"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "abi3-py312 cpu wheel is the primary deliverable (one wheel covers 3.12/3.13/3.14 GIL)"
    - "Two PyPI distributions for two compiled backends (PyPI = one binary wheel per name+version+platform)"
    - "rocm wheel built in-env only, never GitHub Actions (D-06)"

key-files:
  created:
    - crates/catboost-rs-py/pyproject-rocm.toml
    - crates/catboost-rs-py/PACKAGING.md
    - .github/workflows/python-wheels.yml
  modified:
    - crates/catboost-rs-py/pyproject.toml

key-decisions:
  - "abi3-py312 cpu wheel builds, installs, and `import catboost_rs` exposes CatBoostRegressor in a fresh venv — PYAPI-01 cpu deliverable met"
  - "CI builds the cpu/abi3 wheel only; rocm NEVER in Actions (D-06)"
  - "rocm wheel BUILD deferred to gap plan 08-08 (Decision: Option B, generic GPU backend): the facade fit() train path is hardwired to cb_backend::CpuBackend (the only cb-compute::Runtime impl), which is #[cfg(feature=\"cpu\")]-gated — catboost-rs does not compile under --features rocm; 08-08 adds a generic GpuBackend: Runtime over SelectedRuntime (cpu/wgpu/cuda/rocm)"

requirements-completed: [PYAPI-01]

# Metrics
duration: ~20min
completed: 2026-06-23
---

# Phase 8 Plan 07: Per-backend wheel packaging Summary

**The abi3-py312 cpu wheel (`catboost_rs-0.1.0-cp312-abi3-manylinux_2_39_x86_64.whl`) builds, installs into a fresh venv, and `import catboost_rs` exposes `CatBoostRegressor`; the two-distribution layout (cpu `catboost-rs` + rocm `catboost-rs-rocm`, both importing `catboost_rs`, mutually exclusive) is documented, CI builds the cpu wheel only, and the rocm distribution CONFIG (`pyproject-rocm.toml`) is shipped — the rocm wheel BUILD is deferred to gap plan 08-08 (generic GPU backend) because the facade `fit()` train path has no GPU `Runtime` implementation.**

## Status: COMPLETE (in-scope deliverables) — rocm wheel build deferred to 08-08

In-scope packaging deliverables are shipped and validated:

- **abi3-py312 cpu wheel** — built + import-validated.
- **`PACKAGING.md`** — two-distribution model + mutual exclusivity + build/publish split.
- **`.github/workflows/python-wheels.yml`** — CI cpu/abi3 wheel build + import smoke (no rocm in Actions).
- **`pyproject-rocm.toml`** — the rocm *distribution config* (the build-time artifact this plan owns).

The rocm wheel *build* + the GPU `fit`/`predict` smoke are **deferred to gap plan
08-08** (generic GPU backend wiring). **Decision: Option B** — 08-08 will add a
`GpuBackend` implementing `cb_compute::Runtime` over `SelectedRuntime`, generic
across cpu/wgpu/cuda/rocm, calling the existing Phase-7.2 der seam, and
feature-gate `builder.rs`'s backend selection. The rocm wheel cannot compile until
08-08 lands; this plan's responsibility ends at the rocm distribution *config*.

## Performance

- **Duration:** ~20 min
- **Completed:** 2026-06-23
- **Tasks:** 2 (Task 1 complete; Task 2 config complete, rocm wheel build deferred to 08-08)
- **Files created:** 3
- **Files modified:** 1

## Accomplishments (Task 1 — autonomous, verified)

- **abi3-py312 cpu wheel builds:** `maturin build --features cpu --release` →
  `target/wheels/catboost_rs-0.1.0-cp312-abi3-manylinux_2_39_x86_64.whl` (the
  `cp312-abi3` tag proves the limited-API build; one wheel covers 3.12/3.13/3.14 GIL).
- **Fresh-venv import smoke PASSES:** `pip install` the wheel into a clean venv,
  `python -c "import catboost_rs; catboost_rs.CatBoostRegressor"` → `import OK`.
- **`pyproject.toml` finalized** for the cpu distribution: `name="catboost-rs"`,
  `requires-python=">=3.12"`, `[project.optional-dependencies] rocm=["catboost-rs-rocm"]`,
  `[tool.maturin] module-name="catboost_rs"`, `features=["pyo3/extension-module"]`;
  abi3-py312 sourced from the `pyo3` Cargo feature (08-01). Added a clarifying
  comment on the extension-module/abi3 provenance.
- **`PACKAGING.md`** documents the D-08/D-09 two-distribution realization
  (`catboost-rs` cpu + `catboost-rs-rocm` rocm; `catboost-rs[rocm]` → the rocm
  dist; both expose `catboost_rs`; **mutually exclusive** — Pitfall 4), the
  abi3-cpu-primary + free-threaded-wheel-deferred rationale (cross-links
  FREE_THREADING.md), and the build/publish split. Grep gate (`two distribution`,
  `catboost-rs-rocm`, `mutually exclusive`) passes.
- **`.github/workflows/python-wheels.yml`** builds ONLY the cpu/abi3 wheel on
  Python 3.12 (ubuntu), asserts the `abi3` wheel tag, runs a fresh-venv import
  smoke, uploads the wheel artifact, and carries a comment that the rocm wheel is
  NEVER built in Actions (D-06).

## Accomplishments (Task 2 — config artifact, autonomous)

- **`pyproject-rocm.toml`** created: `name="catboost-rs-rocm"`,
  `requires-python=">=3.12"`, `[tool.maturin] module-name="catboost_rs"`,
  `features=["pyo3/extension-module"]`, no `[rocm]` extra (this IS the rocm dist).
  Documents the in-env build command
  `maturin build --no-default-features --features rocm --release`.
- **T-08-21 (cpu leak) re-verified:** `cargo tree -p catboost-rs-py
  --no-default-features --features rocm -e no-dev` is cpu-free (no `cubecl-cpu`);
  `cubecl-cpu` appears in the full tree ONLY via dev-dependencies (the `*_test.rs`
  cb-model/cb-core/cb-train deps), which the wheel build never compiles. The
  build graph links `cubecl-hip` only — the 08-01 mitigation holds.

## Task Commits

1. **Task 1: abi3-py312 cpu wheel config + PACKAGING + CI** — `1b22e0f` (feat)
2. **Task 2 (config artifact): rocm distribution config** — `3834acd` (feat)

## Deviations from Plan

### Rule 4 — Architectural finding (rocm wheel build → deferred to 08-08)

**[Rule 4 - Architectural] The facade `fit()` train path has no GPU `Runtime` implementation; `catboost-rs` does not compile under `--features rocm`. RESOLVED BY DEFERRAL: Decision Option B — owned by gap plan 08-08 (generic GPU backend).**

- **Found during:** Task 2 (attempting the in-env rocm wheel build, gfx1100/ROCm 7.1, HIP toolchain present).
- **Issue:** `maturin build --no-default-features --features rocm --release` fails to compile `catboost-rs`:
  ```
  error[E0432]: unresolved import `cb_backend::CpuBackend`
    --> crates/catboost-rs/src/builder.rs:20
     use cb_backend::CpuBackend;
  note: found an item that was configured out
    --> crates/cb-backend/src/lib.rs:37   pub use cpu_runtime::CpuBackend;  (#[cfg(feature = "cpu")])
  ```
  `crates/catboost-rs/src/builder.rs` `fit()` (lines 20, 346) is hardwired to
  `&CpuBackend`. `CpuBackend` is the **only** type implementing
  `cb_compute::Runtime` (the trait `cb_train::train<R: Runtime>` requires), and it
  is `#[cfg(feature = "cpu")]`-gated in `cb-backend` and internally hard-codes
  `cubecl::cpu::CpuRuntime`. Under `--features rocm` (cpu off) `CpuBackend` does
  not exist, so the facade does not compile and **no rocm wheel can be produced.**
- **Root finding:** The Phase-7 GPU work (7.2–7.6) lives at the `cb-backend`
  kernel / host-light grow-loop layer and is validated by standalone in-env rocm
  tests — it was **never wired through the facade `train()` path**. There is no
  GPU backend implementing `cb-compute::Runtime`. The 08-01 "rocm cpu-free"
  gate validated only the **dependency graph** (`cargo tree`), not actual
  **compilation** of the facade under rocm.
- **Why Rule 4 (not auto-fixed):** Making the rocm wheel build requires
  implementing a GPU `cb-compute::Runtime` for the facade train path — a
  cross-crate change that the user wants done *generically* (wgpu + cuda + rocm),
  not a rocm-only fix. This is an architectural decision, not an inline fix.
- **Files implicated:** `crates/catboost-rs/src/builder.rs` (lines 20, 346),
  `crates/cb-backend/src/lib.rs` (line 37), `crates/cb-backend/src/cpu_runtime.rs`.
- **Resolution — Decision: Option B (deferred to 08-08).** Gap plan **08-08
  (generic GPU backend)** owns this: add a `GpuBackend` implementing
  `cb_compute::Runtime` over `SelectedRuntime`, generic across cpu/wgpu/cuda/rocm,
  calling the existing Phase-7.2 der seam; `builder.rs` selects the backend by
  feature. The rocm wheel *build* + the GPU `fit`/`predict` smoke move to 08-08.
  No code change applied in this plan — 08-07's responsibility ends at the rocm
  distribution *config* (`pyproject-rocm.toml`).

## Human-action gates — resolved by deferral to 08-08

**Task 2 was authored as a `checkpoint:human-action gate="blocking"`** — the
in-env rocm wheel build + GPU import smoke (`fit`/`predict` on gfx1100). Both the
architectural prerequisite and the GPU smoke are **deferred to gap plan 08-08**
(Decision: Option B — generic GPU backend). 08-07 ships the rocm distribution
*config*; 08-08 produces the rocm *wheel* and runs the GPU smoke.

The exact in-env build command 08-08 will use (recorded here, in
`pyproject-rocm.toml`, and in PACKAGING.md) once the generic `GpuBackend` lands:
```bash
cd crates/catboost-rs-py
maturin build --no-default-features --features rocm --release
# then, in a fresh venv on gfx1100:
#   pip install target/wheels/catboost_rs_rocm-*.whl
#   python -c "import catboost_rs; m=catboost_rs.CatBoostRegressor(iterations=5); ..."  # fit/predict smoke
# and confirm the cpu wheel and the rocm wheel are NOT co-installed (mutual exclusivity).
```
This build will NOT compile until 08-08 lands; it was deliberately not attempted here.

## Publish gate (awaiting human authorization)

No wheel was published to PyPI or any index, and nothing was pushed to a remote /
no PR opened (per the plan's autonomy guidance — publish/push require explicit
human authorization). When authorized, the cpu wheel publish is:
```bash
maturin publish --features cpu --release    # cpu distribution catboost-rs
# (rocm wheel: in-env build per above, then publish the catboost-rs-rocm distribution)
```

## Threat Flags

None. The plan's `<threat_model>` (T-08-21..23, T-08-SC) is covered: T-08-21
re-verified cpu-free (no-dev rocm tree), T-08-22 documented in PACKAGING.md
(mutual exclusivity), T-08-23 accepted (rocm in-env only), T-08-SC maturin is the
official PyO3-org tool.

## Next Plan Readiness — gap plan 08-08 (generic GPU backend)

- 08-08 owns the deferred rocm wheel build. Scope (Decision: Option B): add a
  `GpuBackend` implementing `cb_compute::Runtime` over `cb_backend::SelectedRuntime`,
  generic across cpu/wgpu/cuda/rocm, calling the Phase-7.2 der seam; feature-gate
  `crates/catboost-rs/src/builder.rs`'s backend selection (replace the
  unconditional `use cb_backend::CpuBackend` / `&CpuBackend` at lines 20, 346).
- Once 08-08 lands, run the in-env rocm wheel build + GPU smoke (command above)
  using the `pyproject-rocm.toml` config 08-07 ships.
- The T-08-21 cpu-free `--features rocm` dep-tree guarantee (verified here via
  `cargo tree -e no-dev`) means the rocm wheel will link `cubecl-hip` only.

## Self-Check: PASSED

- FOUND: crates/catboost-rs-py/pyproject.toml
- FOUND: crates/catboost-rs-py/PACKAGING.md
- FOUND: crates/catboost-rs-py/pyproject-rocm.toml
- FOUND: .github/workflows/python-wheels.yml
- FOUND: target/wheels/catboost_rs-0.1.0-cp312-abi3-manylinux_2_39_x86_64.whl
- FOUND commit: 1b22e0f (Task 1)
- FOUND commit: 3834acd (Task 2 config)

---
*Phase: 08-python-bindings-dual-api-packaging*
*Completed: 2026-06-23 (in-scope deliverables; rocm wheel build deferred to gap plan 08-08)*
