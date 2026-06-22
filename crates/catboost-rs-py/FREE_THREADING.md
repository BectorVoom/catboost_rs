# Free-threading & `catboost_rs` (PYAPI-06)

This document records how `catboost_rs` satisfies **PYAPI-06 (free-threaded-aware
design)**, why the *free-threaded wheel* is deferred, and the one documented
caveat (the custom-loss callback path).

## TL;DR

- PYAPI-06 is satisfied as a **code property**, not as a shipped artifact:
  - every Python buffer is copied into Rust-owned data **own-before-detach**
    (08-03 / D-11) before the GIL is released, and
  - the module declares `#[pymodule(gil_used = false)]` (PyO3 0.29).
- The **shipped artifact is the abi3-py312 cpu wheel** (one wheel for
  3.12 / 3.13 / 3.14 GIL builds). The **free-threaded wheel is deferred**
  (CONTEXT Deferred Ideas) because `abi3-py312` and free-threading are mutually
  exclusive in PyO3 0.29.
- Runtime validation lives in `tests/test_free_threaded.py`, which **skips** on a
  GIL build and **runs** on a free-threaded interpreter (`python3.13t`).
- **Caveat:** when a Python `custom_loss` / `custom_metric` / `eval_metric`
  callable is supplied, compute **re-enters Python** per der1/der2/eval — that
  one path is documented serialized re-entry, not a fully-detached
  `gil_used=false` claim.

## Why the free-threaded wheel is deferred (abi3 ⊥ free-threaded)

`abi3-py312` (the stable limited API) and PEP 703 free-threading are mutually
exclusive in PyO3 0.29:

- Building `abi3-py312` against a `3.13t` / `3.14t` interpreter emits a warning
  and produces a **non-abi3** wheel — or fails to load on the free-threaded
  runtime — because the free-threaded build uses a new ABI with no limited-API
  equivalent (`abi3t` / PEP 803 is a *future* PyO3 feature, **not in 0.29**).
  (RESEARCH Pitfall 1; pyo3.rs free-threading.)
- Consequence: a free-threaded build would require **per-version** wheels
  (`cp313t`, `cp314t`, …), growing the build matrix beyond this phase's scope.

So the **primary deliverable** is the single `abi3-py312` cpu wheel (PYAPI-01),
and the **free-threaded wheel is deferred** (CONTEXT → Deferred Ideas: "abi3 vs
free-threaded wheel build mechanics"). PYAPI-06 stands on the code property
below, independent of which wheel is shipped.

## How PYAPI-06 is satisfied as a code property

### 1. own-before-detach (D-11, established in 08-03)

Every NumPy / Pandas / Arrow / Polars input is **copied (or quantized) into
Rust-owned `OwnedColumns` before any `Python::detach`**. No borrow into a live
Python buffer (`PyReadonlyArray`, Arrow capsule borrow, …) is ever passed into
the long-running compute closure. Under free threading another thread could
mutate or free a borrowed buffer while detached → UB; copying first removes that
hazard entirely (RESEARCH Pitfall 3).

This own-before-detach discipline is the **single mechanism** that makes
`gil_used=false` sound. It was implemented across the ingest call sites in 08-03;
this phase asserts it under real free threading rather than adding new copying.

### 2. `gil_used = false` (the module-level declaration)

```rust
#[pymodule(gil_used = false)]
fn catboost_rs(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> { /* … */ }
```

Without this flag, importing the module on a free-threaded interpreter would
silently **re-enable the GIL** with a warning. Declaring `gil_used = false` tells
CPython this module is free-threaded-aware, so the interpreter keeps the GIL
disabled — which is only safe *because* of the own-before-detach discipline
above.

### 3. `Model` is `Send + Sync`

The facade `Model` is `Send + Sync` (CLAUDE.md architecture) and `predict` takes
`&self` over owned / quantized data, so concurrent `predict` calls on a single
fitted model are race-free (T-08-19).

## How to validate it

`tests/test_free_threaded.py` spawns ≥ 8 Python threads running concurrent
`fit`/`predict` over both per-thread-private inputs and a single shared,
read-only input array, asserting all results are finite and equal across threads
(no corruption). It **skips** on a GIL build (it is meaningful only under free
threading) and **runs** under a free-threaded interpreter.

```sh
# 1. obtain a free-threaded interpreter (build or install python3.13t)
# 2. create a venv and install maturin into it
python3.13t -m venv .venv-ft
.venv-ft/bin/pip install maturin pytest numpy
# 3. build the extension against the free-threaded interpreter
cd crates/catboost-rs-py
../../.venv-ft/bin/maturin develop --features cpu
# 4. run the test (it will NOT skip on a free-threaded build)
../../.venv-ft/bin/python -m pytest tests/test_free_threaded.py -q
```

On a standard GIL build the same command reports the tests as **skipped** (a
clean skip, never a false pass and never a panic — the Phase-7.5 cpu-skip-guard
lesson).

## Caveat: the custom-loss callback re-enters the GIL (A6)

The one place where compute **must re-enter Python** is the user-supplied
callback path (LOSS-07): a Python `custom_loss` / `custom_metric` /
`eval_metric` callable is wrapped in a Rust struct implementing the facade's
`CustomObjective` / `CustomMetric` traits, and calling it re-attaches the GIL via
**`Python::attach` per der1 / der2 / eval**.

This is **incompatible with a fully-detached compute loop**, so for *that path*
the `gil_used = false` guarantee is documented as **serialized re-entry**, not a
contradiction:

- the re-entry is per-call `Python::attach` (the callback runs under the GIL it
  re-acquires), so the Python callable itself is never called concurrently
  without GIL protection;
- it is **not a fixture-reachable default path** — the default losses
  (RMSE / Logloss / etc.) run fully detached over owned data; only an explicit
  Python callback opts into re-entry.

(See RESEARCH "LOSS-07 Python callback bridge" and Assumption A6; threat
register T-08-20 disposition = *accept*, documented caveat.)
