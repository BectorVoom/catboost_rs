# Phase 8: Python Bindings, Dual API & Packaging - Context

**Gathered:** 2026-06-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver a **PyO3 binding crate** (new, e.g. `crates/catboost-rs-py`, `cdylib`) that wraps the existing `catboost-rs` facade into a Python package whose surface **mirrors upstream CatBoost**. The package is simultaneously scikit-learn-compatible and CatBoost-native through a single unified class hierarchy. It accepts NumPy / Pandas / Arrow / Polars input, maps typed `thiserror` errors to specific Python exceptions, is free-threaded-aware, and is distributed as per-backend wheels (cpu + rocm) for Python ≥ 3.12.

**Locked direction (user, carried forward):**
- **Rust side keeps the existing `CatBoostBuilder`** (Builder pattern, already shipped in Phase 4). Python is a *separate idiom* — a CatBoost mirror — NOT a transliteration of the Rust Builder.
- **Python surface mirrors CatBoost**: parameter names, default values, and class layout follow upstream CatBoost.
- PyO3 + maturin; **no C API** (PROJECT constraint); `abi3-py312`, Python ≥ 3.12.

**Requirements in scope:** PYAPI-01, PYAPI-02, PYAPI-03, PYAPI-04, PYAPI-05, PYAPI-06 (see REQUIREMENTS.md — not duplicated here).

**Not this phase (do not pursue):** new training/algorithm capability (Phases 3–7 own the core); GPU kernel work (Phase 7); R/CLI surfaces (PROJECT out-of-scope).
</domain>

<decisions>
## Implementation Decisions

### Dual-API layering (PYAPI-02, PYAPI-03)
- **D-01:** **Mirror upstream — one unified class hierarchy.** The sklearn methods (`fit`/`predict`/`predict_proba`/`score`/`get_params`/`set_params`) live directly ON the `CatBoostClassifier` / `CatBoostRegressor` / `CatBoostRanker` classes, exactly like real CatBoost. There is **no separate sklearn wrapper module** — the CatBoost-native classes *are* the sklearn-compatible estimators.
- **D-02:** **All classes implemented as PyO3 `#[pyclass]` types** — `CatBoostClassifier`, `CatBoostRegressor`, `CatBoostRanker`, `Pool`. No pure-Python shim package on top of the compiled core. Consequence (must be planned for): `get_params` / `set_params`, `__sklearn_tags__`, `clone`-ability, `NotFittedError`, and the full param signatures all have to be implemented **in Rust** via PyO3, not in Python glue.

### scikit-learn compliance (PYAPI-02)
- **D-03:** **Core estimator contract is a hard test gate**, BUT scoped (revised during discussion — see note). The structural `check_estimator` checks MUST pass: cloneability, `get_params`/`set_params` round-trip, no work/mutation in `__init__`, `NotFittedError` before fit, predict-shape, usable inside an sklearn `Pipeline`.
- **D-04:** **The dtype/contiguity-related `check_estimator` checks are documented skips.** Rationale: D-13 makes input strictly float32/contiguous, which is incompatible with `check_estimator` feeding float64 / non-contiguous arrays. This is a deliberate, documented allowlist of skipped checks — NOT a silent gap. (This **revises** an earlier "full pass, zero skips" stance once the conflict with D-13 surfaced.)

### Unsupported-parameter policy (PYAPI-03, PYAPI-05)
- **D-05:** **Reject unsupported parameters with a clear, typed error.** A user migrating from CatBoost who passes a real-but-unimplemented param (GPU-only knobs, unimplemented losses, niche options) gets an explicit exception — no silent wrong results, parity gaps surface loudly.
- **D-06:** **Validation fires at `fit()` time, never in `__init__`.** `__init__` stores all params verbatim (required to satisfy the `check_estimator` "no work in `__init__`" contract, D-03). The unsupported/unknown-param error raises when `fit()` runs — still pre-training, just later than construction.
- **D-07:** **Maintain a registry of the FULL CatBoost parameter vocabulary.** Error messages distinguish two cases: (a) *known CatBoost param, not yet implemented in catboost-rs* (flagged as a parity gap), vs (b) *unrecognized param* (likely typo → suggest closest match). This is explicit build scope — the registry must mirror upstream CatBoost's param list and be maintained as parity grows.

### Packaging & distribution (PYAPI-01)
- **D-08:** **One user-facing PyPI distribution, `catboost-rs`**, with backend selected via an extras-style selector (`catboost-rs` = cpu default, `catboost-rs[rocm]` = rocm GPU). User intent = single discovery name, backend as an install option. *(Research must pin the actual mechanism — PyPI extras pull Python deps, not alternate compiled binaries; the likely realization is a `catboost-rs` cpu/meta package + a `[rocm]` extra that depends on a rocm binary package. The intent is captured; the maturin/PyPI mechanism is a research item — see Deferred/Research.)*
- **D-09:** **Import name is `catboost_rs`** — no shadowing of the real `catboost` package. Honest, collision-free, coexists side-by-side with a real CatBoost install. Migration costs a one-line import change. (Shadow-`catboost` and opt-in-shim alternatives were explicitly rejected.)

### Input ingestion & GIL handling (PYAPI-04, PYAPI-06)
- **D-10:** **Zero-copy where safe, copy otherwise.** Borrow C-contiguous, correct-dtype (float32) NumPy / Arrow buffers zero-copy *while the GIL guarantees liveness*. Copy + convert only the non-contiguous / wrong-dtype / Pandas-object / nullable-Arrow cases. Honors the PROJECT memory-efficiency constraint.
- **D-11:** **Own/quantize before releasing the GIL** (PYAPI-06 free-threaded-aware). The GIL is released only after the input data is owned (copied) or quantized — never hold a borrow into a live Python buffer across a GIL release, since another thread could mutate/free it. Long compute (training, batch apply) runs only on owned/quantized data.
- **D-12:** **Strict input validation — reject mismatches with actionable messages.** Wrong dtype (float64), non-contiguous layout, ambiguous object columns (string columns without `cat_features` set), and unsupported nullable types are rejected with messages telling the user how to fix it (e.g. "pass float32", "np.ascontiguousarray(X)"). Predictable behavior vs the ≤10⁻⁵ oracle bar; no silent precision-changing coercion.
- **D-13:** Cross-cutting consequence of D-12: because input must be float32 + contiguous, the dtype-related `check_estimator` checks become documented skips (D-04). These two decisions are linked — do not "fix" one without revisiting the other.

### Claude's Discretion
- Exact PyO3 module/crate name and internal file layout.
- Exception class taxonomy under PYAPI-05 (specific names/hierarchy) — as long as it maps typed `thiserror` variants to specific Python exceptions with actionable messages (e.g. a `CatBoostParameterError` for D-05/D-07).
- Which concrete sklearn `check_estimator` checks land on the documented-skip allowlist (must be enumerated and justified at implementation time).
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope & requirements
- `.planning/ROADMAP.md` §"Phase 8: Python Bindings, Dual API & Packaging" — goal, mode (mvp), dependency (Phase 7), requirement list.
- `.planning/REQUIREMENTS.md` — PYAPI-01 … PYAPI-06 full text (the locked *what*).
- `.planning/PROJECT.md` — constraints (PyO3+maturin, no C API, Python ≥ 3.12, abi3, per-backend wheels, memory efficiency, dual sklearn+CatBoost-native API), Key Decisions table.

### The Rust surface being wrapped
- `crates/catboost-rs/src/lib.rs` — the published facade: `CatBoostBuilder`, `Model`, `Pool`, `IngestSource`/`OwnedColumns`, and re-exported enums (`Loss`, `LeafMethod`, `EScoreFunction`, `EBootstrapType`, `PredictionType`, `FeatureImportanceType`). This is the exact API the PyO3 layer binds to.
- `crates/catboost-rs/src/builder.rs` — `CatBoostBuilder` setters + `fit(&pool) -> Result<Model, CatBoostError>` (param surface to mirror in Python kwargs).
- `crates/catboost-rs/src/model.rs` — `Model`: `predict`/`predict_proba`/`predict_with`, `save_cbm`/`load_cbm`/`save_json`/`load_json`, `shap_values`, `feature_importance`.
- `crates/catboost-rs/src/error.rs` — `CatBoostError` typed `thiserror` enum (source for the PYAPI-05 exception mapping).
- `crates/cb-data/src/ingest.rs` (`IngestSource`, `OwnedColumns`) + `crates/cb-data` `Pool` — the existing ingestion seam the NumPy/Pandas/Arrow/Polars adapters feed into.
- `Cargo.toml` (workspace root) — backend feature wiring (`cpu`/`rocm`/`wgpu`/`cuda`), clippy restriction-lint policy (`unwrap`/`expect`/`panic`/`indexing_slicing` denied), centralized dependency pins.

### Upstream parity reference (oracle + param vocabulary)
- `catboost-master/catboost/python-package/catboost/core.py` — upstream `CatBoostClassifier`/`Regressor`/`Ranker`/`Pool` Python API to mirror (class layout, method signatures, sklearn methods).
- `catboost-master/catboost/python-package/catboost/_catboost.pyx` — upstream Cython core; reference for the .pyx-over-C++ split (we chose all-in-PyO3 instead, D-02, but signatures/behavior are the parity target).
- Upstream param documentation/source for the **full CatBoost parameter vocabulary** registry (D-07) — to be located precisely during research (params are defined across `core.py` and the C++ options layer).
</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `catboost-rs` facade (`CatBoostBuilder` + `Model` + `Pool`): the binding wraps THIS crate, not the internal `cb-*` crates directly. Keeps the unsafe/internal boundary intact.
- `cb-data::ingest` (`IngestSource`, `OwnedColumns`): existing ingestion path — the NumPy/Pandas/Arrow/Polars adapters should converge onto this rather than inventing a new ingestion seam.
- `CatBoostError` typed enum: direct source for the PYAPI-05 `thiserror`→Python-exception mapping (one Rust variant → one specific Python exception).

### Established Patterns
- **Builder is Rust-only.** Do not expose the Builder to Python; the Python kwargs→fit mapping is the CatBoost-mirror idiom (D-01).
- **Workspace clippy gate** (`unwrap`/`expect`/`panic`/`indexing_slicing` = deny; test code exempt in-code). The new PyO3 crate must opt into `[lints] workspace = true` and obey these — PyO3 glue code included.
- **Backend feature unification landmine** (from prior-phase memory): never let a backend feature leak `cpu` onto a `--features rocm` build via unification. The per-backend wheel feature wiring must respect the same discipline used in `cb-backend`.
- `abi3-py312` + free-threaded interact awkwardly (a known tension) — the wheel build strategy must reconcile abi3 with PYAPI-06's free-threaded-aware design (research item).

### Integration Points
- New `cdylib` crate (e.g. `crates/catboost-rs-py`) depending on `catboost-rs` (facade) + PyO3 + (transitively) the selected backend.
- maturin build → per-backend wheels (cpu, rocm) under one `catboost-rs` distribution (D-08).
- Python-side packaging artifacts (pyproject/maturin config) — first Python packaging surface in the repo (none exists yet outside test fixture generators).
</code_context>

<specifics>
## Specific Ideas

- User's framing verbatim: "design rust native (build pattern). python wrapper is catboost mirror." → Rust keeps the Builder; Python is a faithful CatBoost mirror (D-01).
- The all-PyO3-`#[pyclass]` choice (D-02) intentionally diverges from upstream's pure-Python-over-Cython split — single compiled artifact preferred over a Python shim package.
- Strictness-over-forgiveness leaning throughout: reject unsupported params (D-05), reject bad dtype/layout (D-12) — honesty and oracle-bar predictability prioritized over maximum migration leniency.
</specifics>

<deferred>
## Deferred Ideas

- **Ranker → sklearn mapping** — sklearn has no native ranker base estimator; how `CatBoostRanker` presents under sklearn conventions is a researcher/planner decision, not a user-vision call.
- **`.cbm` model-format interop with upstream CatBoost** — whether a `.cbm` produced by real CatBoost loads in catboost-rs (and vice versa) — left to research/planning; not raised as a Phase-8 user requirement.
- **abi3 vs free-threaded wheel build mechanics** — abi3-py312 and PEP 703 free-threading interact awkwardly; the concrete maturin build matrix is a research item (feeds D-08/PYAPI-06).
- **Exact PyPI extras → binary-backend mechanism (D-08)** — PyPI extras pull Python deps, not alternate compiled binaries; research must pin the realization (meta/cpu package + `[rocm]` extra depending on a rocm binary package, or equivalent). Intent locked; mechanism open.
- **`wgpu`/`cuda` wheels** — PYAPI-01 requires only cpu + rocm minimum this phase; other backends are future packaging work.
- **Shadow-`catboost` import / opt-in compat shim** — explicitly rejected for this phase (D-09); noted in case a future "zero-change migration" push wants to revisit.

</deferred>

---

*Phase: 8-python-bindings-dual-api-packaging*
*Context gathered: 2026-06-21*
