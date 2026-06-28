# Phase 8: Python Bindings, Dual API & Packaging - Research

**Researched:** 2026-06-21
**Domain:** PyO3 Python bindings, maturin packaging, scikit-learn estimator contract, zero-copy NumPy/Arrow/Polars ingestion, free-threaded CPython
**Confidence:** HIGH (versions/APIs verified via crates.io + official PyO3/sklearn/maturin docs; one genuinely-open packaging mechanism flagged LOW)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Mirror upstream — one unified class hierarchy. sklearn methods (`fit`/`predict`/`predict_proba`/`score`/`get_params`/`set_params`) live directly ON `CatBoostClassifier`/`CatBoostRegressor`/`CatBoostRanker`, exactly like real CatBoost. **No separate sklearn wrapper module** — the CatBoost-native classes *are* the sklearn estimators.
- **D-02:** All classes are PyO3 `#[pyclass]` types — `CatBoostClassifier`, `CatBoostRegressor`, `CatBoostRanker`, `Pool`. No pure-Python shim on top of the compiled core. Consequence: `get_params`/`set_params`, `__sklearn_tags__`, clone-ability, `NotFittedError`, full param signatures all implemented **in Rust** via PyO3.
- **D-03:** Core estimator contract is a hard test gate, scoped. The structural `check_estimator` checks MUST pass: cloneability, `get_params`/`set_params` round-trip, no work/mutation in `__init__`, `NotFittedError` before fit, predict-shape, usable inside an sklearn `Pipeline`.
- **D-04:** The dtype/contiguity-related `check_estimator` checks are **documented skips** (linked to D-13). A deliberate, documented allowlist — NOT a silent gap.
- **D-05:** Reject unsupported parameters with a clear, typed error. No silent wrong results.
- **D-06:** Validation fires at `fit()` time, never in `__init__`. `__init__` stores all params verbatim (to satisfy "no work in `__init__`", D-03). Unknown/unsupported-param error raises when `fit()` runs.
- **D-07:** Maintain a registry of the FULL CatBoost parameter vocabulary. Errors distinguish (a) known CatBoost param not yet implemented (parity gap) vs (b) unrecognized param (typo → suggest closest match). Registry mirrors upstream and grows as parity grows.
- **D-08:** One user-facing PyPI distribution `catboost-rs`, backend selected via an extras-style selector (`catboost-rs` = cpu default, `catboost-rs[rocm]` = rocm GPU). **Mechanism is a research item** (PyPI extras pull Python deps, not alternate compiled binaries; likely a cpu/meta package + `[rocm]` extra depending on a rocm binary package).
- **D-09:** Import name is `catboost_rs` — no shadowing of the real `catboost`. Coexists side-by-side. Migration = one-line import change.
- **D-10:** Zero-copy where safe, copy otherwise. Borrow C-contiguous, float32 NumPy/Arrow buffers zero-copy *while the GIL guarantees liveness*. Copy+convert only non-contiguous / wrong-dtype / Pandas-object / nullable-Arrow cases.
- **D-11:** Own/quantize before releasing the GIL (PYAPI-06). GIL released only after input is owned (copied) or quantized — never hold a borrow into a live Python buffer across a GIL release. Long compute runs only on owned/quantized data.
- **D-12:** Strict input validation — reject mismatches with actionable messages (wrong dtype, non-contiguous, ambiguous object columns, unsupported nullable types). No silent precision-changing coercion.
- **D-13:** Cross-cutting consequence of D-12: because input must be float32 + contiguous, the dtype-related `check_estimator` checks become documented skips (D-04). Do not "fix" one without revisiting the other.

### Claude's Discretion
- Exact PyO3 module/crate name and internal file layout.
- Exception class taxonomy under PYAPI-05 (specific names/hierarchy) — must map typed `thiserror` variants to specific Python exceptions with actionable messages (e.g. `CatBoostParameterError` for D-05/D-07).
- Which concrete sklearn `check_estimator` checks land on the documented-skip allowlist (must be enumerated and justified at implementation time).

### Deferred Ideas (OUT OF SCOPE)
- **Ranker → sklearn mapping** — sklearn has no native ranker base estimator; how `CatBoostRanker` presents under sklearn conventions is a researcher/planner decision.
- **`.cbm` model-format interop with upstream CatBoost** (round-trip with real CatBoost) — left to research/planning; not a Phase-8 user requirement.
- **abi3 vs free-threaded wheel build mechanics** — concrete maturin build matrix is a research item.
- **Exact PyPI extras → binary-backend mechanism (D-08)** — research must pin the realization.
- **`wgpu`/`cuda` wheels** — only cpu + rocm required this phase.
- **Shadow-`catboost` import / opt-in compat shim** — explicitly rejected (D-09).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PYAPI-01 | PyO3 + maturin per-backend wheels (`cpu`+`rocm`), `abi3-py312`, Python ≥ 3.12 | Standard Stack (pyo3 0.29 `abi3-py312`, maturin ≥1.9.4); Packaging section pins per-backend wheel mechanism + the abi3↔free-threaded conflict resolution |
| PYAPI-02 | scikit-learn API (`fit`/`predict`/`predict_proba`/`score`/`get_params`/`set_params`); passes `check_estimator` | sklearn Estimator Contract section (the exact contract, the Rust-in-`#[pyclass]` realization, the D-04 documented-skip allowlist) |
| PYAPI-03 | CatBoost-native API — `Pool`, `CatBoostClassifier`/`Regressor`/`Ranker`, full param-name parity + defaults | Parameter Vocabulary section (full upstream param list extracted from `core.py`); D-07 registry design |
| PYAPI-04 | NumPy/Pandas/Arrow/Polars input with dtype/contiguity validation | Input Ingestion section (`numpy` 0.29 `PyReadonlyArray`, `pyo3-arrow` 0.19, the existing `IngestSource` seam, validation rules) |
| PYAPI-05 | Typed `thiserror` → specific Python exception mapping with actionable messages | Error Mapping section (`create_exception!`, `From<CatBoostError> for PyErr`) |
| PYAPI-06 | Free-threaded-aware design — no GIL reliance for buffer safety (copy/quantize under GIL before release) | Free-Threading section (`Python::detach`, own-before-detach, the abi3↔nogil conflict, `gil_used` module flag) |
</phase_requirements>

## Summary

This phase wraps the existing **`catboost-rs` facade crate** (the published `CatBoostBuilder` + `Model` + `Pool` + `CatBoostError` surface) in a new PyO3 `cdylib` crate that presents a CatBoost-mirror Python API. All four user-facing classes (`Pool`, `CatBoostClassifier`, `CatBoostRegressor`, `CatBoostRanker`) are `#[pyclass]` types (D-02), so the sklearn contract (`get_params`/`set_params`, `__sklearn_tags__`, clone-ability, `NotFittedError`) is implemented in Rust, not Python glue. Input ingestion (NumPy/Pandas/Arrow/Polars) converges onto the existing `cb_data::ingest::IngestSource` seam with new borrowed/zero-copy adapters; typed `CatBoostError` variants map one-to-one onto specific PyO3 exceptions.

Two **hard landmines** dominate the packaging design and must be planned for explicitly:

1. **abi3 and free-threaded CPython are mutually exclusive in PyO3 0.29.** `[VERIFIED: pyo3.rs free-threading docs]` The free-threaded build uses a new ABI with no limited-API equivalent; abi3 wheels **cannot load** on a free-threaded interpreter, and the `abi3` feature is *ignored with a warning* when building for free-threaded Python. PYAPI-01 (`abi3-py312`) and PYAPI-06 (free-threaded-aware) therefore cannot be satisfied by a single wheel. The resolution: ship an **abi3-py312 wheel** as the default deliverable (one wheel covers 3.12/3.13/3.14 GIL builds), and treat free-threaded support as a *design property of the Rust code* (own/quantize-before-detach, `gil_used=false` module flag) that is *validated* on a separate version-specific 3.13t/3.14t build but not necessarily shipped this phase. PYAPI-06 is satisfiable as a **code-correctness requirement** even if a free-threaded *wheel* is deferred.

2. **PyPI extras cannot select alternate compiled binaries.** `[CITED: PyPI/maturin packaging model]` An extra (`catboost-rs[rocm]`) pulls additional *Python package dependencies*; it cannot swap the compiled `.so` inside the same wheel. Under one distribution name+version+platform tag PyPI accepts exactly one binary wheel. The realization of D-08 is therefore a **multi-distribution** layout: `catboost-rs` (cpu wheel, the default) + a separately-named binary distribution `catboost-rs-rocm` (rocm wheel), with the `[rocm]` extra on `catboost-rs` declaring a dependency on `catboost-rs-rocm`. Both wheels expose the same `catboost_rs` import name and must never be installed simultaneously (document the conflict).

**Primary recommendation:** Create `crates/catboost-rs-py` (`cdylib`, `crate-type=["cdylib","rlib"]`), depending on the `catboost-rs` facade + `pyo3 0.29` (`abi3-py312`) + `numpy 0.29` + `pyo3-arrow 0.19`. Implement all four classes as `#[pyclass]`. Ship an **abi3-py312 cpu wheel** as the primary artifact; build a parallel rocm wheel under `catboost-rs-rocm` via a maturin feature-driven build matrix. Treat free-threading as a code-design property validated on a 3.13t build. Gate every external package behind the legitimacy audit below.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Python class surface (`#[pyclass]`) | PyO3 binding crate (`catboost-rs-py`) | — | D-02: no Python shim; classes are compiled |
| sklearn contract (`get_params`/tags/clone) | PyO3 binding crate (Rust) | — | D-02: implemented in Rust, not Python |
| Param vocabulary registry (D-07) | PyO3 binding crate (Rust) | upstream `core.py` (parity source) | Registry mirrors upstream; validated at `fit()` (D-06) |
| Kwargs → Builder mapping | PyO3 binding crate | `catboost-rs` facade (`CatBoostBuilder`) | Builder stays Rust-only (D-01); binding translates |
| Train / predict / explain | `catboost-rs` facade → `cb-*` crates | — | Existing core; binding never reaches internal crates directly |
| Input ingestion (NumPy/Pandas/Arrow/Polars) | PyO3 binding crate (adapters) | `cb_data::ingest::IngestSource` → `Pool` | New borrowed adapters converge on the existing seam (D-10) |
| GIL/buffer safety (own-before-detach) | PyO3 binding crate | — | D-11/PYAPI-06; only the binding touches Python buffers |
| Error mapping (thiserror → PyErr) | PyO3 binding crate | `catboost-rs::CatBoostError` (source) | One Rust variant → one Python exception (PYAPI-05) |
| Backend feature propagation (cpu/rocm) | `cb-backend` feature table | `catboost-rs-py` Cargo features → maturin `--features` | Existing discipline; binding forwards, never pins cpu |
| Wheel build / packaging | maturin + `pyproject.toml` | CI matrix | abi3 cpu wheel + separate rocm distribution |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `pyo3` | 0.29.0 | Rust↔Python bindings, `#[pyclass]`, `abi3-py312`, free-threading primitives | The de-facto Rust/Python FFI; latest stable, has the free-threading + abi3 features this phase needs `[VERIFIED: crates.io pyo3=0.29.0]` |
| `maturin` | ≥ 1.9.4 | Build + package the `cdylib` into wheels driven by Cargo features | Official PyO3 packaging tool; ≥1.9.4 sets `PYO3_BUILD_EXTENSION_MODULE` automatically `[CITED: pyo3.rs building-and-distribution]` |
| `numpy` (rust-numpy) | 0.29.0 | `PyReadonlyArray` zero-copy NumPy borrow, dtype/contiguity introspection | PyO3-version-locked rust-numpy; `PyReadonlyArray` borrows pointer+metadata while Python keeps ownership `[VERIFIED: crates.io numpy=0.29.0]` |
| `pyo3-arrow` | 0.19.0 | Arrow C Data / PyCapsule zero-copy ingest for Arrow + Polars + buffer-protocol objects | Implements the Arrow PyCapsule interface; auto zero-copy conversion of numpy/memoryview/Arrow into Rust `arrow` arrays `[VERIFIED: crates.io pyo3-arrow=0.19.0]` |
| `arrow` | 59.0.0 | Rust Arrow arrays (already a workspace dep; the target type `pyo3-arrow` yields) | Already pinned in `cb-data` ingest (`crates/cb-data/src/ingest/arrow.rs`) `[VERIFIED: workspace Cargo.toml]` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `scikit-learn` | 1.9.0 | `check_estimator` / `parametrize_with_checks` test gate (pytest dev-dep) | Test-time only — the contract validator (D-03) `[CITED: scikit-learn.org/stable develop]` |
| `pytest` | latest | Drive the Python test suite over a `maturin develop` build | Standard PyO3 test harness `[CITED: maturin.rs]` |
| `polars` | 0.54.4 | Polars ingestion (already a workspace dep) | Polars `DataFrame` → Arrow C-stream → `pyo3-arrow` path `[VERIFIED: workspace Cargo.toml]` |
| `thiserror` | 2.0.18 | Source of the typed `CatBoostError` (already published by the facade) | Binding adds a `CatBoostParameterError`-style enum + `From … for PyErr` `[VERIFIED: workspace Cargo.toml]` |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `pyo3-arrow` 0.19 for Arrow/Polars | Hand-roll Arrow C Data Interface via raw FFI | pyo3-arrow already implements the PyCapsule protocol + buffer-protocol zero-copy; hand-rolling reinvents documented, error-prone FFI (Don't Hand-Roll) |
| abi3-py312 single wheel | Per-version wheels (cp312/cp313/cp314 + 3.13t/3.14t free-threaded) | abi3 gives one wheel for all GIL builds but *cannot* cover free-threaded; per-version is the only path to a shippable free-threaded wheel (deferred) |
| All-`#[pyclass]` (D-02 locked) | Pure-Python shim over a thin compiled core (upstream's Cython split) | D-02 locks all-in-PyO3; noted only because the sklearn contract is harder in Rust (manual `get_params` introspection) |

**Installation (Rust side — `crates/catboost-rs-py/Cargo.toml`):**
```toml
[dependencies]
pyo3 = { version = "0.29.0", features = ["abi3-py312", "extension-module"] }
numpy = "0.29.0"
pyo3-arrow = "0.19.0"
catboost-rs = { path = "../catboost-rs" }
cb-data = { path = "../cb-data" }   # for IngestSource / Pool / OwnedColumns
thiserror = { workspace = true }
```
```toml
[lib]
crate-type = ["cdylib", "rlib"]   # cdylib for the wheel; rlib so Rust tests can link

[features]
default = ["cpu"]
cpu  = ["catboost-rs/<cpu-passthrough>"]   # forward to cb-backend cpu; NEVER pin cpu unconditionally
rocm = ["catboost-rs/<rocm-passthrough>"]
```

**Build (Python side — `pyproject.toml`):**
```toml
[build-system]
requires = ["maturin>=1.9.4,<2.0"]
build-backend = "maturin"

[project]
name = "catboost-rs"
requires-python = ">=3.12"

[project.optional-dependencies]
rocm = ["catboost-rs-rocm"]    # extra pulls the separate rocm binary distribution (D-08 realization)

[tool.maturin]
module-name = "catboost_rs"    # import name (D-09)
features = ["pyo3/extension-module"]
```

**Version verification performed:**
- `cargo search pyo3` → `0.29.0` `[VERIFIED: crates.io 2026-06-21]`
- `cargo search numpy` → `0.29.0` (rust-numpy) `[VERIFIED: crates.io 2026-06-21]`
- `cargo search pyo3-arrow` → `0.19.0` `[VERIFIED: crates.io 2026-06-21]`
- PyO3 0.29.0 released 2026-06-11, adds Python 3.15 beta support `[CITED: github.com/PyO3/pyo3/releases]`
- maturin ≥1.9.4 sets `PYO3_BUILD_EXTENSION_MODULE` `[CITED: pyo3.rs building-and-distribution]`
- scikit-learn 1.9.0 is current docs line `[CITED: scikit-learn.org/stable]`

## Package Legitimacy Audit

> All four Rust crates are mainstream PyO3-ecosystem packages with long histories and high download counts; legitimacy verified via crates.io presence + known authoritative source (PyO3 org / official docs). The `package-legitimacy check` seam is npm/PyPI-oriented; for crates the verification is crates.io age + GitHub org reputation + official-docs cross-reference.

| Package | Registry | Age / Source | Source Repo | Verdict | Disposition |
|---------|----------|--------------|-------------|---------|-------------|
| `pyo3` | crates.io | mature, 0.29.0 (2026-06-11) | github.com/PyO3/pyo3 | OK | Approved — official PyO3 org |
| `numpy` (rust-numpy) | crates.io | mature, 0.29.0, PyO3-version-locked | github.com/PyO3/rust-numpy | OK | Approved — official PyO3 org |
| `pyo3-arrow` | crates.io | active (updated 2026-02), 0.19.0 | github.com/kylebarron/arro3 | OK | Approved — established Arrow-ecosystem author (geoarrow/arro3) |
| `maturin` | crates.io / PyPI | mature, ≥1.9.4 | github.com/PyO3/maturin | OK | Approved — official PyO3 org |
| `scikit-learn` | PyPI | mature, 1.9.0 | github.com/scikit-learn | OK | Approved (test dep) |

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

*Note: `pyo3-arrow` is the only non-PyO3-org dependency; its author (kylebarron) maintains the widely-used geoarrow/arro3 stack. The planner should add a single `checkpoint:human-verify` before first `cargo add pyo3-arrow` to confirm 0.19.x is the intended pin, given it is the newest/least-canonical of the five.*

## Architecture Patterns

### System Architecture Diagram

```text
  Python user code (sklearn Pipeline / CatBoost-style)
        │  X: np.ndarray / pd.DataFrame / pa.Table / pl.DataFrame ; kwargs
        ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  catboost_rs  (the cdylib — crates/catboost-rs-py)           │
 │                                                             │
 │  #[pyclass] CatBoostClassifier / Regressor / Ranker / Pool  │
 │     __init__  → store kwargs verbatim (NO work, D-06)       │
 │     get_params/set_params/__sklearn_tags__/clone  (Rust)    │
 │                                                             │
 │   fit(X, y):                                                │
 │     1. param-registry validate (D-05/D-07) ── unknown ─┐    │
 │     2. ingest X  ──► [GIL held] ─────────────────┐     │    │
 │          NumPy  → PyReadonlyArray (borrow)        │     │    │
 │          Pandas → via numpy / arrow               │     ▼    ▼
 │          Arrow  → pyo3-arrow PyCapsule (borrow)   │  CatBoostParameterError
 │          Polars → Arrow C-stream → pyo3-arrow     │   (specific PyErr, PYAPI-05)
 │     3. dtype/contiguity validate (D-12) ──bad──► CatBoostValueError
 │     4. OWN/COPY-or-QUANTIZE under GIL (D-10/D-11) │
 │     5. Python::detach() ◄── release GIL ──────────┘
 │            │ owned/quantized data only
 │            ▼
 │     IngestSource::into_pool() → cb_data::Pool               │
 │            ▼                                                │
 │     catboost-rs::CatBoostBuilder ...setters... .fit(&pool)  │
 │            ▼                                                │
 │     catboost-rs::Model  (store on the #[pyclass], "_"-attr) │
 │                                                             │
 │   predict/predict_proba/score → Model::predict* → np.array │
 │   CatBoostError ── From ──► specific PyErr (PYAPI-05)       │
 └─────────────────────────────────────────────────────────────┘
        │ depends on (rlib)              │ Cargo feature cpu/rocm
        ▼                                 ▼
   catboost-rs facade            cb-backend (SelectedRuntime)
   (Builder/Model/Pool/Error)    cpu→cubecl/cpu  rocm→cubecl/hip
        │
        ▼  cb-train / cb-model / cb-compute / cb-data / cb-core
```

### Recommended Project Structure
```
crates/catboost-rs-py/
├── Cargo.toml              # cdylib+rlib; pyo3/numpy/pyo3-arrow/catboost-rs deps; cpu/rocm features
├── pyproject.toml          # maturin build-backend; module-name=catboost_rs; [rocm] extra
├── src/
│   ├── lib.rs              # #[pymodule] catboost_rs; register classes + exceptions; gil_used flag
│   ├── estimator.rs        # shared #[pyclass] estimator base logic (params store, get/set, tags, clone, NotFitted)
│   ├── classifier.rs       # CatBoostClassifier #[pyclass]
│   ├── regressor.rs        # CatBoostRegressor #[pyclass]
│   ├── ranker.rs           # CatBoostRanker #[pyclass]
│   ├── pool.rs             # Pool #[pyclass]
│   ├── params.rs           # full CatBoost param-vocabulary registry (D-07) + kwargs→Builder map
│   ├── ingest_py.rs        # NumPy/Pandas/Arrow/Polars → IngestSource adapters (D-10/D-11/D-12)
│   ├── errors.rs           # create_exception! taxonomy + From<CatBoostError> for PyErr (PYAPI-05)
│   └── *_test.rs           # Rust-side unit tests (source/test separation rule)
└── tests/                  # Python pytest suite (sklearn check_estimator, oracle parity)
    ├── test_check_estimator.py
    ├── test_oracle_parity.py
    ├── test_ingestion.py
    └── test_free_threaded.py
python/                     # (optional) type stubs (.pyi) for the compiled module
```

### Pattern 1: `#[pyclass]` estimator storing params verbatim, validating at `fit()`
**What:** `__init__` writes every kwarg into a params map with no transformation; `get_params` reads it back; `fit` is where validation + Builder construction happen.
**When to use:** Every estimator class (D-02/D-03/D-06).
```rust
// Source: sklearn develop contract + PyO3 0.29 #[pyclass] guide
#[pyclass(subclass)]
pub struct CatBoostClassifier {
    params: BTreeMap<String, Py<PyAny>>, // stored verbatim (D-06: no work in __init__)
    model: Option<catboost_rs::Model>,   // the fitted "model_" attr; None ⇒ NotFitted
}

#[pymethods]
impl CatBoostClassifier {
    #[new]
    #[pyo3(signature = (**kwargs))]
    fn new(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        // store kwargs verbatim; DO NOT validate or coerce here (D-06, sklearn "no work in __init__")
        Ok(Self { params: collect_params(kwargs)?, model: None })
    }

    fn get_params(&self, py: Python<'_>, deep: Option<bool>) -> PyResult<Py<PyDict>> {
        // must round-trip exactly with set_params (sklearn contract)
        params_to_pydict(py, &self.params)
    }

    fn set_params(&mut self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> { /* merge */ Ok(()) }

    fn fit(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<()> {
        validate_params(&self.params)?;                 // D-05/D-07: unknown/unsupported → CatBoostParameterError
        let pool = ingest_to_pool(py, x, Some(y))?;     // D-10/D-11: own/quantize under GIL, then detach for compute
        let model = build_and_fit(&self.params, &pool)?; // maps to CatBoostBuilder; CatBoostError → PyErr
        self.model = Some(model);                        // sets the fitted "_" state
        Ok(())
    }
}
```

### Pattern 2: Own-then-detach for free-threaded safety (PYAPI-06 / D-11)
**What:** Borrow the Python buffer only long enough to copy/quantize into owned Rust memory, then `Python::detach` (PyO3 0.29's renamed `allow_threads`) for the long compute.
```rust
// Source: pyo3.rs free-threading guide (0.28.3)
fn ingest_to_pool(py: Python<'_>, x: &Bound<'_, PyAny>, y: Option<&Bound<'_, PyAny>>) -> PyResult<Pool> {
    // --- GIL HELD: buffer is alive only here ---
    let arr: PyReadonlyArray2<f32> = x.extract()?;      // borrow; validates ndarray
    validate_contiguous_f32(&arr)?;                     // D-12 strict; else CatBoostValueError
    let owned: OwnedColumns = copy_into_owned(&arr);    // D-11: OWN before any detach
    // --- now safe to release: no live borrow into Python memory ---
    let pool = owned.into_pool()?;                      // IngestSource seam
    Ok(pool)
}
// the heavy fit() compute then runs under py.detach(|| builder.fit(&pool))
```
> **PyO3 0.29 rename:** `allow_threads` → `detach`, and the `Python<'py>`/`Bound` "GIL held" semantics become "thread attached" under free-threading. `[VERIFIED: pyo3.rs free-threading]`

### Pattern 3: Typed error → specific Python exception (PYAPI-05)
```rust
// Source: PyO3 create_exception! + From<E> for PyErr idiom
create_exception!(catboost_rs, CatBoostError,        pyo3::exceptions::PyException);
create_exception!(catboost_rs, CatBoostParameterError, CatBoostError); // D-05/D-07
create_exception!(catboost_rs, CatBoostValueError,     CatBoostError); // D-12 dtype/layout
create_exception!(catboost_rs, NotFittedError,         CatBoostError); // sklearn parity name

impl From<catboost_rs::CatBoostError> for PyErr {
    fn from(e: catboost_rs::CatBoostError) -> Self {
        match e {
            catboost_rs::CatBoostError::FeatureMismatch(m) => CatBoostValueError::new_err(m),
            catboost_rs::CatBoostError::Train(c)           => CatBoostError::new_err(c.to_string()),
            catboost_rs::CatBoostError::Model(m)           => CatBoostError::new_err(m.to_string()),
            catboost_rs::CatBoostError::Io(io)             => PyIOError::new_err(io.to_string()),
            other => CatBoostError::new_err(other.to_string()),
        }
    }
}
```
> sklearn's real `NotFittedError` subclasses `ValueError`. For `check_estimator` to recognize the not-fitted path, the Python wrapper test should either raise sklearn's own `sklearn.exceptions.NotFittedError` or the binding's `NotFittedError` must subclass `PyValueError` (planner decision under PYAPI-05 discretion).

### Pattern 4: Zero-copy Arrow / Polars via pyo3-arrow PyCapsule
```rust
// Source: pyo3-arrow docs — PyCapsule interface, zero-copy into arrow crate
fn ingest_arrow(table: pyo3_arrow::PyTable) -> PyResult<OwnedColumns> {
    // pyo3-arrow gives an arrow-rs RecordBatch/Array zero-copy via the C Data Interface.
    // Polars DataFrame exposes the same PyCapsule (__arrow_c_stream__), so one path serves both.
    // Then converge onto the EXISTING cb_data::ingest::arrow adapter (ArrowColumns).
    let cols = pytable_to_arrow_columns(table)?;  // validate dtype=Float32, no nulls (D-12)
    Ok(cols.into_owned())                          // OWN before detach (D-11)
}
```

### Anti-Patterns to Avoid
- **Validating/coercing in `__init__`.** Breaks the sklearn "no work in `__init__`" check (D-03/D-06). Store verbatim; validate in `fit`.
- **Holding a `PyReadonlyArray` borrow across `py.detach()`.** Under free-threading another thread can mutate/free it (D-11). Always own first.
- **Pinning `cpu` unconditionally in the binding's Cargo features.** Feature unification forces cpu onto a `--features rocm` build, defeating a cpu-free rocm wheel (prior-phase landmine; `cb-backend` discipline).
- **One wheel for abi3 *and* free-threaded.** Impossible in PyO3 0.29 — abi3 is ignored on free-threaded builds. Separate wheels (one deferred).
- **A pure-Python shim package on top.** Contradicts D-02 (all `#[pyclass]`).
- **Shadowing the `catboost` import name.** D-09 — import name is `catboost_rs`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Arrow C Data Interface / PyCapsule FFI | Raw `extern "C"` schema/array import | `pyo3-arrow` 0.19 | Documented, zero-copy, handles capsule lifetime + Polars `__arrow_c_stream__` |
| NumPy buffer borrow + dtype/strides | Manual buffer-protocol parsing | `numpy::PyReadonlyArray` | Borrow + contiguity/dtype introspection with Python keeping ownership |
| sklearn contract test | Custom conformance asserts | `sklearn.utils.estimator_checks.parametrize_with_checks` / `check_estimator` | The authoritative validator; reproduces GridSearchCV/Pipeline expectations |
| Wheel building / ABI tagging | Manual `setup.py` + cargo invocation | `maturin` | Sets `PYO3_BUILD_EXTENSION_MODULE`, correct abi3/platform tags |
| Python exception hierarchy | Returning string errors | `create_exception!` + `From for PyErr` | Real Python exception types, catchable, with `__cause__` chaining |

**Key insight:** The PyO3 ecosystem has first-class, version-locked solutions for every FFI boundary this phase touches (NumPy borrow, Arrow capsule, wheel tagging, exception creation). The project's memory-efficiency constraint is *better* served by these than by hand-rolled FFI, because they implement the documented zero-copy paths correctly.

## Common Pitfalls

### Pitfall 1: abi3 silently ignored on free-threaded builds
**What goes wrong:** Building with `abi3-py312` against a 3.13t/3.14t interpreter emits a warning and produces a *non-abi3* wheel — or fails to load on the free-threaded runtime. PYAPI-01 (abi3) and PYAPI-06 (free-threaded) collide.
**Why it happens:** The free-threaded build uses a new ABI with no limited-API equivalent; abi3 wheels cannot load there `[VERIFIED: pyo3.rs free-threading]`.
**How to avoid:** Ship the **abi3-py312 cpu wheel** as the primary artifact (covers 3.12/3.13/3.14 GIL builds with one wheel). Treat free-threading as a *code property* (own-before-detach, `#[pymodule(gil_used = false)]`), validated on a separate version-specific 3.13t build, with the free-threaded *wheel* explicitly deferred per CONTEXT (it is in Deferred Ideas).
**Warning signs:** maturin warning "abi3 ignored for free-threaded"; `ImportError` on a `python3.13t` interpreter.

### Pitfall 2: `get_params`/`set_params` round-trip failure breaks GridSearchCV
**What goes wrong:** `check_estimator` clones the estimator via `get_params` → `__init__(**params)` and expects identical behavior. If `__init__` renames, drops, or coerces a kwarg, the round-trip fails.
**Why it happens:** Storing a transformed copy instead of the verbatim kwarg.
**How to avoid:** D-06 — `__init__` stores kwargs verbatim into the params map; `get_params` returns them unchanged; every `__init__` keyword is a stored attribute. `[CITED: scikit-learn develop]`
**Warning signs:** `check_estimator` failures in `check_get_params_invariance` / `check_set_params`.

### Pitfall 3: Holding a Python buffer borrow across the compute / GIL release
**What goes wrong:** A `PyReadonlyArray` (or Arrow capsule borrow) held while `py.detach()` runs the long fit; under free-threading another thread mutates/frees it → UB.
**Why it happens:** Treating the borrow as owned.
**How to avoid:** D-11 — copy/quantize into `OwnedColumns` *before* any detach; never pass a borrow into the compute closure.
**Warning signs:** sporadic crashes/corruption only under free-threaded multi-thread tests.

### Pitfall 4: PyPI rejects two binary wheels under one name (D-08)
**What goes wrong:** Attempting to publish a cpu and a rocm wheel both as `catboost-rs` on the same platform tag — PyPI accepts only one binary wheel per name+version+platform.
**Why it happens:** Extras pull Python deps, not alternate `.so`s.
**How to avoid:** Two distributions — `catboost-rs` (cpu) + `catboost-rs-rocm` (rocm), with `catboost-rs[rocm]` → depends on `catboost-rs-rocm`; both expose `catboost_rs`; document mutual-exclusivity. `[CITED: PyPI binary-distribution model]`
**Warning signs:** Twine "file already exists" / filename-collision on upload.

### Pitfall 5: `__sklearn_tags__` missing or stale for sklearn ≥1.6
**What goes wrong:** sklearn 1.6+ replaced the old `_get_tags()` dict with the `__sklearn_tags__()` dataclass (`Tags`). A binding implementing the old API fails modern `check_estimator`.
**Why it happens:** Training-data examples predate the 1.6 tags refactor.
**How to avoid:** Implement `__sklearn_tags__` returning a `Tags` instance (estimator_type classifier/regressor, target/input tags). For the Ranker, decide the tag presentation (Deferred: ranker→sklearn mapping). `[CITED: scikit-learn 1.9 develop]`
**Warning signs:** `AttributeError: __sklearn_tags__` or tag-driven checks skipped/failing.

## Runtime State Inventory

> Phase 8 is greenfield (a new binding crate + Python packaging). No rename/refactor of existing runtime state. The only "state" is build/packaging config introduced fresh:

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — no datastores keyed on a renamed string | None |
| Live service config | None | None |
| OS-registered state | None | None |
| Secrets/env vars | PyPI publish token (CI), `PYO3_*` build env vars (`PYO3_USE_ABI3_FORWARD_COMPATIBILITY` for 3.14+) — new, not a rename | Configure in CI; not a migration |
| Build artifacts | New: `target/wheels/*.whl`, `crates/catboost-rs-py/pyproject.toml`, maturin metadata — all net-new | Create fresh; nothing stale to clean |

**Nothing found in stored-data / live-service / OS-registered categories** — verified: this phase adds a crate and Python packaging, touching no existing persisted or registered state.

## Code Examples

### Full param vocabulary registry (D-07) — source of truth
The full upstream `CatBoostClassifier.__init__` keyword list (~130 params) extracted from the vendored source `catboost-master/catboost/python-package/catboost/core.py:5333` is the registry seed. Representative subset (the registry must mirror the *entire* list and tag each as IMPLEMENTED / KNOWN-NOT-YET / UNKNOWN):
```text
iterations, learning_rate, depth, l2_leaf_reg, model_size_reg, rsm, loss_function,
border_count, feature_border_type, per_float_feature_quantization, input_borders,
output_borders, fold_permutation_block, od_pval, od_wait, od_type, nan_mode,
counter_calc_method, leaf_estimation_iterations, leaf_estimation_method, thread_count,
random_seed, use_best_model, best_model_min_trees, verbose, silent, logging_level,
metric_period, ctr_leaf_count_limit, store_all_simple_ctr, max_ctr_complexity, has_time,
allow_const_label, target_border, classes_count, class_weights, auto_class_weights,
class_names, one_hot_max_size, random_strength, random_score_type, name, ignored_features,
train_dir, custom_loss, custom_metric, eval_metric, bagging_temperature, save_snapshot,
snapshot_file, snapshot_interval, fold_len_multiplier, used_ram_limit, gpu_ram_part,
pinned_memory_size, allow_writing_files, final_ctr_computation_mode, approx_on_full_history,
boosting_type, simple_ctr, combinations_ctr, per_feature_ctr, ctr_description,
ctr_target_border_count, task_type, device_config, devices, bootstrap_type, subsample,
mvs_reg, sampling_unit, sampling_frequency, ..., max_depth, n_estimators, num_boost_round,
num_trees, colsample_bylevel, random_state, reg_lambda, objective, eta, max_bin,
scale_pos_weight, ..., grow_policy, min_data_in_leaf, min_child_samples, max_leaves,
num_leaves, score_function, leaf_estimation_backtracking, ctr_history_unit,
monotone_constraints, feature_weights, penalties_coefficient, first_feature_use_penalties,
per_object_feature_penalties, model_shrink_rate, model_shrink_mode, langevin,
diffusion_temperature, posterior_sampling, boost_from_average, text_features, tokenizers,
dictionaries, feature_calcers, text_processing, embedding_features, callback, eval_fraction,
fixed_binary_splits
```
Note the **sklearn-alias params** (`max_depth`↔`depth`, `n_estimators`/`num_trees`/`num_boost_round`↔`iterations`, `random_state`↔`random_seed`, `reg_lambda`↔`l2_leaf_reg`, `objective`↔`loss_function`, `eta`↔`learning_rate`, `max_bin`↔`border_count`, `colsample_bylevel`↔`rsm`) — full parity requires accepting these aliases too. `[VERIFIED: catboost-master/.../core.py:5333]`

### Upstream `Pool.__init__` surface to mirror (PYAPI-03)
```text
Pool(data, label=None, cat_features=None, text_features=None, embedding_features=None,
     embedding_features_data=None, column_description=None, pairs=None, graph=None,
     delimiter='\t', has_header=False, ignore_csv_quoting=False, weight=None,
     group_id=None, group_weight=None, subgroup_id=None, pairs_weight=None, baseline=None,
     timestamp=None, feature_names=None, feature_tags=None, thread_count=-1, ...)
```
`[VERIFIED: catboost-master/.../core.py:608]` Maps onto the existing `cb_data::Pool` accessors (`float_features`/`cat_features`/`text_features`/`embedding_features`/`label`/`weights`/`group_id`/`subgroup_id`/`pairs`/`baseline`) — the binding's `Pool` builds an `OwnedColumns` and calls `into_pool()`.

### Existing Rust facade surface the binding wraps
```rust
// catboost-rs facade (crates/catboost-rs/src/lib.rs) — the EXACT bind target:
CatBoostBuilder::new()
    .loss(Loss).iterations(usize).depth(usize).learning_rate(f64)
    .l2_leaf_reg(f64).random_strength(f64).leaf_method(LeafMethod)
    .bootstrap_type(EBootstrapType).subsample(f64).bagging_temperature(f32)
    .random_seed(u64).border_count(usize).score_function(EScoreFunction)
    .custom_objective(Arc<dyn CustomObjective>).custom_metric(Arc<dyn CustomMetric>)
    .fit(&Pool) -> Result<Model, CatBoostError>;
// Model: predict / predict_proba / predict_with / shap_values / feature_importance
//        / save_cbm / load_cbm / save_json / load_json
```
`[VERIFIED: crates/catboost-rs/src/{builder.rs,model.rs}]` The kwargs→Builder map (D-07 registry) covers exactly these setters; any upstream param without a corresponding setter is a KNOWN-NOT-YET parity gap (D-07 case (a)).

### LOSS-07 Python callback bridge (deferred from Phase 6.4)
`CustomObjective`/`CustomMetric` are `Arc<dyn …>` Rust traits in the facade (Phase 6.4, D-09). The Python `custom_loss`/`custom_metric`/`eval_metric` callback wraps a `Py<PyAny>` callable in a Rust struct implementing those traits — calling back into Python (re-attaching the GIL via `Python::attach`) per der1/der2/eval. This is the one place compute must *re-enter* Python, so it is incompatible with a fully-detached compute loop and free-threaded `gil_used=false` claims; plan a documented caveat. `[VERIFIED: ROADMAP Phase 6.4 / facade builder.rs custom_objective]`

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `Python::allow_threads(...)` | `Python::detach(...)` | PyO3 0.29 | Rename; same GIL-release semantics, free-threading-aware naming |
| `_get_tags()` dict | `__sklearn_tags__()` → `Tags` dataclass | sklearn 1.6 | Bindings must implement the new method or modern `check_estimator` fails |
| `GILPool` / `Python::acquire_gil` | `Python::attach` / `Bound<'py, T>` | PyO3 0.21–0.23 | The `Bound` API is the only supported handle; gil-refs removed |
| Per-version cpXY wheels only | abi3 single wheel for GIL builds | stable for years | One abi3-py312 wheel covers 3.12/3.13/3.14 (GIL); free-threaded still needs per-version |
| Hand-rolled Arrow FFI | Arrow PyCapsule interface (`pyo3-arrow`) | 2024–2026 | Zero-copy Arrow/Polars/buffer-protocol via one crate |

**Deprecated/outdated:**
- `pyo3::Python::acquire_gil` / gil-ref API: removed; use `Python::attach` + `Bound`/`Py`.
- sklearn `_get_tags`/`_more_tags`: superseded by `__sklearn_tags__` (1.6+).
- abi3t (PEP 803 single free-threaded+GIL wheel): *future* (PyO3 tracking for 3.15+); **not available in 0.29** — do not plan around it this phase.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The `catboost-rs` facade exposes enough setters to cover the *core* sklearn methods for clf/reg/ranker without new core work; unimplemented upstream params become D-07 KNOWN-NOT-YET errors rather than blockers | Architecture / Param registry | If a sklearn-contract-required behavior (e.g. `predict_proba` for a loss) is missing from the facade, a small core/facade addition may be needed — surfaces at planning when mapping each method |
| A2 | An abi3-py312 cpu wheel is an acceptable *primary* deliverable for PYAPI-01, with the free-threaded wheel deferred (it is in CONTEXT Deferred Ideas) and PYAPI-06 satisfied as a code property | Packaging / Free-threading | If PYAPI-06 is read as requiring a *shipped* free-threaded wheel, the build matrix grows (per-version 3.13t/3.14t wheels) — confirm scope with user at discuss/plan |
| A3 | The D-08 realization is a two-distribution layout (`catboost-rs` + `catboost-rs-rocm` with a `[rocm]` extra dependency); no single-wheel mechanism exists for swapping the compiled backend | Packaging | If user expects literally one wheel selectable at install, that is not achievable on PyPI; the two-distribution model is the standard realization (cf. torch cpu/cu121 wheels) |
| A4 | rocm wheel build is feasible in CI without GPU hardware at build time (compile-only; cubecl/hip links against HIP libs, not a live GPU) — runtime needs the GPU, build does not | Validation / Packaging | If the rocm wheel cannot be *built* without a GPU present, the CI matrix needs an AMD builder (the project already runs rocm in-env per memory) |
| A5 | `NotFittedError` presentation: subclassing `PyValueError` (or raising sklearn's own) satisfies the not-fitted `check_estimator` path | Error mapping | Minor — adjustable; the exact base class is PYAPI-05 discretion |
| A6 | The custom-objective Python callback (LOSS-07) re-entering the GIL during compute is acceptable and does not need to coexist with a `gil_used=false` free-threaded claim for the *callback* path | Code Examples | If free-threaded custom-loss is required, the callback path needs per-call `Python::attach` with documented serialization — a known caveat, not a blocker |

**Note:** A1–A6 are LOW/MEDIUM-risk planning assumptions; none contradicts a locked decision. Confirm A2/A3 (packaging scope) with the user during planning since they shape the wheel matrix.

## Open Questions (RESOLVED)

1. **Free-threaded wheel — ship or defer? — RESOLVED.**
   - What we know: abi3 and free-threaded are mutually exclusive in PyO3 0.29; CONTEXT lists "abi3 vs free-threaded wheel build mechanics" under Deferred Ideas.
   - **Resolution:** PYAPI-06 is satisfied as a CODE property (own-before-detach at every ingest call site + `#[pymodule(gil_used=false)]`), validated on a 3.13t build; the free-threaded *wheel* is DEFERRED per CONTEXT Deferred Ideas (abi3 ⊥ free-threaded in PyO3 0.29). Addressed in plan 08-06 (gil_used flag + multi-thread buffer-safety test; the abi3 cpu wheel remains the shipped artifact, plan 08-07).

2. **Ranker → sklearn presentation (CONTEXT Deferred). — RESOLVED.**
   - What we know: sklearn has no native ranker base estimator / mixin.
   - **Resolution:** `CatBoostRanker` presents with a regressor-like tag set (continuous score output via `__sklearn_tags__` estimator_type="regressor") AND is EXCLUDED from the structural `check_estimator` gate with a documented justification (no native sklearn ranker contract to satisfy). Addressed in plan 08-05 (Task 1 decides the tag set; Task 2 enumerates the ranker exclusion in the gate).

3. **`.cbm` interop with upstream CatBoost (CONTEXT Deferred). — RESOLVED.**
   - What we know: the facade already loads upstream 1.2.10 `.cbm`/`.json` (Phase 4, MODEL-01).
   - **Resolution:** expose a `load_model(path)` classmethod on the native estimators wrapping `catboost_rs::Model::load_cbm`; interop already works at the Rust layer — minimal Python-side work, no new research blocker. Addressed in plan 08-04 (Task 2 uses `load_model` to drive the deterministic oracle-parity test against the stored reference vector).

4. **CI builder for the rocm wheel. — RESOLVED.**
   - What we know: project runs rocm in-env on gfx1100 (memory); GitHub Actions never runs rocm tests.
   - **Resolution:** the rocm wheel builds in-env ONLY (gfx1100/ROCm 7.1, manual checkpoint); the cpu/abi3 wheel is the only GitHub-CI wheel. Consistent with the Phase-7 precedent (D-06: rocm never in Actions). Addressed in plan 08-07 (Task 1 cpu wheel in CI; Task 2 rocm wheel in-env human-action checkpoint).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust / cargo | Building the cdylib | ✓ | rustc 1.96.0 | — |
| Python (system) | maturin develop / pytest | ✓ | 3.12.3 | — |
| Python 3.13 (.venv) | abi3 covers it; free-threaded test needs 3.13**t** | ✓ (3.13 GIL) | 3.13 (`.venv`) | free-threaded 3.13t must be built/installed separately for PYAPI-06 test |
| maturin | Wheel build + `maturin develop` | ✗ | — | `pip install maturin>=1.9.4` (or `uv tool install maturin`) — install step in plan |
| scikit-learn | `check_estimator` test gate | ✗ | — | `pip install scikit-learn` in the test venv (Wave 0) |
| catboost==1.2.10 | Oracle-parity fixtures (≤1e-5) | ✓ (per memory, in `.venv`) | 1.2.10 | reuse existing offline fixtures; do not require live import for unit tests |
| numpy / pandas / pyarrow / polars (Python) | Ingestion tests (PYAPI-04) | ✗ (need to confirm in test venv) | — | `pip install numpy pandas pyarrow polars` in the test venv |
| HIP / ROCm toolchain | Building the rocm wheel | ✓ (in-env, gfx1100, ROCm 7.1 per memory) | ROCm 7.1 | rocm wheel built in-env only |

**Missing dependencies with no fallback:** none (all installable).
**Missing dependencies with fallback:**
- `maturin`, `scikit-learn`, Python `numpy/pandas/pyarrow/polars`, and a free-threaded `python3.13t` are not yet present — all are pip/build-installable; the plan must include a Wave-0 test-venv setup step.

## Validation Architecture

> `workflow.nyquist_validation` not disabled in config → section included.

### Test Framework
| Property | Value |
|----------|-------|
| Rust-side framework | built-in `#[test]` in `*_test.rs` files (source/test separation rule) |
| Python-side framework | `pytest` over a `maturin develop` build |
| Config file | none yet — Wave 0 creates `crates/catboost-rs-py/pyproject.toml` + a test venv |
| Quick run (Rust) | `cargo test -p catboost-rs-py` |
| Quick run (Python) | `maturin develop && pytest crates/catboost-rs-py/tests -x` |
| Full suite | `cargo test --workspace` + `pytest crates/catboost-rs-py/tests` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PYAPI-01 | abi3-py312 cpu wheel builds + imports on 3.12/3.13 | smoke | `maturin build --features cpu && pip install <whl> && python -c "import catboost_rs"` | ❌ Wave 0 |
| PYAPI-01 | rocm wheel builds in-env | smoke | `maturin build --no-default-features --features rocm` (in-env) | ❌ Wave 0 |
| PYAPI-02 | passes structural `check_estimator` (clone, get/set round-trip, no-work-in-init, NotFitted, predict-shape, Pipeline) | integration | `pytest tests/test_check_estimator.py -x` (with documented-skip allowlist per D-04) | ❌ Wave 0 |
| PYAPI-03 | full param-name parity + defaults vs upstream `core.py` | unit | `pytest tests/test_params.py::test_signature_matches_upstream` | ❌ Wave 0 |
| PYAPI-03 | clf/reg/ranker fit→predict produces correct shapes | unit | `pytest tests/test_native_api.py` | ❌ Wave 0 |
| PYAPI-03 | oracle parity ≤1e-5 vs catboost 1.2.10 fixtures (Python surface) | integration | `pytest tests/test_oracle_parity.py -x` (reuse offline fixtures) | ❌ Wave 0 |
| PYAPI-04 | NumPy/Pandas/Arrow/Polars zero-copy ingest + dtype/contiguity reject | unit | `pytest tests/test_ingestion.py` | ❌ Wave 0 |
| PYAPI-05 | typed errors map to specific catchable exceptions w/ actionable messages | unit | `pytest tests/test_errors.py` | ❌ Wave 0 |
| PYAPI-06 | own-before-detach buffer safety under free threads | integration | `python3.13t -m pytest tests/test_free_threaded.py` (multi-thread fit/predict, no corruption) | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test -p catboost-rs-py` (Rust unit) + `pytest -x` on the touched test file.
- **Per wave merge:** `maturin develop && pytest crates/catboost-rs-py/tests` + `cargo test --workspace`.
- **Phase gate:** full Python suite green (including `check_estimator` with documented skips) + abi3 cpu wheel + rocm wheel build smoke + oracle parity ≤1e-5 before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `crates/catboost-rs-py/pyproject.toml` — maturin build config (module-name `catboost_rs`, abi3-py312, features)
- [ ] Test venv with `maturin>=1.9.4`, `scikit-learn`, `numpy`, `pandas`, `pyarrow`, `polars` (+ optional `python3.13t` for PYAPI-06)
- [ ] `tests/conftest.py` — shared fixtures (toy datasets, reuse of existing offline oracle fixtures)
- [ ] `tests/test_check_estimator.py` — the structural gate + documented-skip allowlist (D-04, enumerated)
- [ ] `tests/test_oracle_parity.py` — Python-surface ≤1e-5 vs the existing catboost 1.2.10 fixtures
- [ ] `tests/test_free_threaded.py` — buffer-safety under a free-threaded interpreter
- [ ] Param-vocabulary registry seed extracted from `catboost-master/.../core.py` (D-07)

## Security Domain

> `security_enforcement` not disabled → included. This is an FFI binding surface; the relevant threats are input-validation and memory-safety at the Python boundary.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No auth surface (library binding) |
| V3 Session Management | no | No sessions |
| V4 Access Control | no | No access control surface |
| V5 Input Validation | yes | Strict dtype/contiguity/shape validation at `fit()` (D-12); typed rejection, never silent coercion; param-registry rejects unknown kwargs (D-05) |
| V6 Cryptography | no | No crypto |
| V12 / Memory safety (FFI) | yes | Own-before-detach (D-11) — never hold a Python-buffer borrow across GIL release; no `unwrap`/`expect`/`panic` across the boundary (workspace clippy gate); all FFI returns typed `PyErr` |

### Known Threat Patterns for PyO3 binding

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Use-after-free of a borrowed Python buffer across GIL release (free-threaded) | Tampering / DoS | Copy/quantize into `OwnedColumns` before `py.detach()` (D-11); validated by `test_free_threaded.py` |
| Out-of-bounds read from a feature-count mismatch | Tampering | `CatBoostError::FeatureMismatch` → `CatBoostValueError` (already enforced in the facade) |
| Panic crossing the FFI boundary (process abort) | DoS | Workspace clippy denies `unwrap`/`expect`/`panic`/`indexing_slicing`; every fallible path returns `PyResult` |
| Silent precision coercion (float64→float32) changing results | Tampering (correctness) | D-12 strict reject with actionable message; no silent coercion |
| Untrusted `.cbm`/`.json` model load | Tampering | Facade already returns typed `ModelError` on malformed input (Phase 4, never panics) |

## Sources

### Primary (HIGH confidence)
- crates.io via `cargo search` — `pyo3=0.29.0`, `numpy=0.29.0`, `pyo3-arrow=0.19.0` (verified 2026-06-21)
- `catboost-master/catboost/python-package/catboost/core.py` — upstream `Pool.__init__` (line 608) + `CatBoostClassifier.__init__` (line 5333) full parameter vocabulary (vendored oracle source)
- `crates/catboost-rs/src/{lib.rs,builder.rs,model.rs,error.rs}` — exact facade bind target
- `crates/cb-data/src/ingest/mod.rs` + `pool.rs` — `IngestSource`/`OwnedColumns`/`Pool` seam
- `crates/cb-backend/Cargo.toml` — backend feature wiring (cpu/wgpu/cuda/rocm; cubecl/hip)

### Secondary (MEDIUM confidence)
- pyo3.rs free-threading guide (v0.28.3) — `Python::detach`/`attach`, abi3↔free-threaded conflict, `gil_used` module flag, own-before-detach pattern
- pyo3.rs building-and-distribution / features — abi3-pyXY features, `extension-module`, maturin integration
- scikit-learn.org/stable develop + `check_estimator` — estimator contract, `__sklearn_tags__`, trailing-underscore, `n_features_in_`, no-work-in-init
- github.com/PyO3/pyo3/releases — 0.29.0 (2026-06-11) release notes
- maturin.rs distribution — `--features`/`--no-default-features` build flags, `pyproject.toml [tool.maturin]`

### Tertiary (LOW confidence)
- WebSearch summaries on the D-08 PyPI extras-vs-binary-backend mechanism (no single official doc; the two-distribution realization is inferred from the documented PyPI binary-wheel model and torch's cpu/cuXXX precedent — flagged A3)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all four crate versions verified on crates.io; APIs cross-referenced to official docs
- Architecture: HIGH — bind target is the existing facade, read directly from source
- Packaging (abi3/free-threaded): HIGH on the conflict (official PyO3 docs), MEDIUM on the deferral scope (A2)
- Packaging (D-08 per-backend distribution): LOW-MEDIUM — no single official doc; standard realization inferred (A3)
- sklearn contract: HIGH — official develop docs + check_estimator reference
- Pitfalls: HIGH — each tied to an official-doc or vendored-source fact

**Research date:** 2026-06-21
**Valid until:** 2026-07-21 (PyO3/maturin/sklearn move fast — re-verify versions if planning slips past ~30 days; PyO3 0.30 and PEP 803 abi3t may land)

## RESEARCH COMPLETE

**Phase:** 8 - Python Bindings, Dual API & Packaging
**Confidence:** HIGH

### Key Findings
- **Stack pinned & verified:** `pyo3 0.29.0` (`abi3-py312`,`extension-module`), `numpy 0.29.0` (rust-numpy, `PyReadonlyArray`), `pyo3-arrow 0.19.0` (Arrow/Polars/buffer-protocol zero-copy), `maturin ≥1.9.4`, `scikit-learn 1.9.0` (test gate) — all confirmed on crates.io / official docs.
- **Landmine 1 (abi3 ⊥ free-threaded):** Mutually exclusive in PyO3 0.29 — abi3 is *ignored* on free-threaded builds. Resolution: ship abi3-py312 cpu wheel (covers 3.12/3.13/3.14 GIL); satisfy PYAPI-06 as a code property (own-before-detach + `gil_used=false`) validated on 3.13t; defer the free-threaded *wheel* (it is in CONTEXT Deferred).
- **Landmine 2 (D-08 packaging):** PyPI cannot put two binary wheels under one name. Realization = two distributions: `catboost-rs` (cpu) + `catboost-rs-rocm` (rocm), `[rocm]` extra depends on the latter; both import as `catboost_rs`.
- **Bind target is the existing `catboost-rs` facade** (`CatBoostBuilder`/`Model`/`Pool`/`CatBoostError`) + the `cb_data::ingest::IngestSource` seam — no internal-crate reach-through; new borrowed adapters converge on the existing ingestion path.
- **Param registry seed extracted** from vendored `core.py` (~130 `CatBoostClassifier` params + full `Pool` signature), including sklearn-alias params (`max_depth`/`n_estimators`/`random_state`/`reg_lambda`/`eta`/`max_bin`/`colsample_bylevel`) — the D-07 registry source of truth.

### File Created
`.planning/phases/08-python-bindings-dual-api-packaging/08-RESEARCH.md`

### Confidence Assessment
| Area | Level | Reason |
|------|-------|--------|
| Standard Stack | HIGH | Versions verified on crates.io; APIs from official docs |
| Architecture | HIGH | Bind target read directly from facade source |
| Pitfalls | HIGH | Each tied to official-doc or vendored-source fact |
| Packaging (D-08) | LOW-MEDIUM | No single official doc; two-distribution realization inferred (A3) |

### Open Questions
- Free-threaded wheel: ship or defer? (recommend defer the wheel, satisfy PYAPI-06 as code property — confirm with user)
- Ranker → sklearn `__sklearn_tags__` presentation (no native sklearn ranker base)
- rocm wheel CI builder (recommend in-env build where HIP toolchain lives)

### Ready for Planning
Research complete. Planner can create PLAN.md files; recommend a `checkpoint:human-verify` on (a) `pyo3-arrow 0.19` pin and (b) the A2/A3 packaging-scope assumptions before locking the wheel matrix.
