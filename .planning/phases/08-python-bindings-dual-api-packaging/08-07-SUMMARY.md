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
  - "rocm wheel BUILD is BLOCKED (Rule 4 architectural): the facade fit() train path is hardwired to cb_backend::CpuBackend (the only cb-compute::Runtime impl), which is #[cfg(feature=\"cpu\")]-gated — catboost-rs does not compile under --features rocm"

requirements-completed: []
requirements-partial: [PYAPI-01]

# Metrics
duration: ~20min
completed: 2026-06-23
---

# Phase 8 Plan 07: Per-backend wheel packaging Summary

**The abi3-py312 cpu wheel (`catboost_rs-0.1.0-cp312-abi3-manylinux_2_39_x86_64.whl`) builds, installs into a fresh venv, and `import catboost_rs` exposes `CatBoostRegressor`; the two-distribution layout (cpu `catboost-rs` + rocm `catboost-rs-rocm`, both importing `catboost_rs`, mutually exclusive) is documented and CI builds the cpu wheel only — but the rocm wheel BUILD is blocked because the facade `fit()` train path has no GPU `Runtime` implementation (architectural, Rule 4 / Task-2 blocking checkpoint).**

## Status: CHECKPOINT (Task 2 blocking)

- **Task 1 — COMPLETE & committed** (abi3 cpu wheel + PACKAGING.md + CI workflow).
- **Task 2 — config artifact COMPLETE & committed** (`pyproject-rocm.toml`); the **rocm wheel build is BLOCKED** by a facade architectural gap (details below) and additionally requires the in-env GPU import smoke (the original human-action gate).

## Performance

- **Duration:** ~20 min
- **Completed:** 2026-06-23
- **Tasks:** 2 (1 complete, 1 blocked at checkpoint)
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

### Rule 4 — Architectural blocker (rocm wheel build)

**[Rule 4 - Architectural] The facade `fit()` train path has no GPU `Runtime` implementation; `catboost-rs` does not compile under `--features rocm`.**

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
- **Why Rule 4 (not auto-fixed):** Making the rocm wheel build requires either
  (a) implementing a GPU `cb-compute::Runtime` for the facade train path (a
  cross-crate change explicitly scoped to Phase 7 GPU work, "GPU kernel work
  (Phase 7)" is NOT this phase per CONTEXT), or (b) a deliberate decision to ship
  a rocm wheel whose `fit()` still orchestrates via a CPU `Runtime` (host-light)
  with GPU kernels reachable only at the cb-backend layer — which needs
  `cb-backend` to expose a `Runtime` impl under non-cpu features. Both are
  architectural decisions, not an inline fix.
- **Files implicated:** `crates/catboost-rs/src/builder.rs` (lines 20, 346),
  `crates/cb-backend/src/lib.rs` (line 37), `crates/cb-backend/src/cpu_runtime.rs`.
- **Status:** Surfaced as the Task-2 blocking checkpoint (below). No code change
  applied — the decision is the user's.

## Authentication / Human-action gates

**Task 2 was authored as a `checkpoint:human-action gate="blocking"`** — the
in-env rocm wheel build + GPU import smoke (`fit`/`predict` on gfx1100) is the
human verification step because it needs the live GPU runtime. The blocking
checkpoint now carries TWO items:

1. **(Architectural — Rule 4)** Decide how the facade train path reaches a GPU
   backend so `catboost-rs` compiles under `--features rocm` (the rocm wheel
   cannot build until this is decided/implemented). Likely a follow-up plan /
   Phase-7 GPU-facade-wiring task, not packaging scope.
2. **(Original human-action — GPU smoke)** Once (1) is resolved and the rocm wheel
   builds, run the in-env import + fit/predict smoke on gfx1100 and confirm a cpu
   wheel and the rocm wheel are not co-installed.

The exact in-env build command (recorded in `pyproject-rocm.toml` and PACKAGING.md):
```bash
cd crates/catboost-rs-py
maturin build --no-default-features --features rocm --release
```

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
*Completed (Task 1) / Checkpoint (Task 2): 2026-06-23*
