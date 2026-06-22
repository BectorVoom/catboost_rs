"""PYAPI-06 free-threaded buffer-safety validation (08-06).

This module proves the **own-before-detach** discipline (08-03 / D-11) holds
under *real* free threading: concurrent ``fit``/``predict`` from many Python
threads — over both per-thread-private inputs and a single shared read-only
input array — must produce finite, cross-thread-consistent results with no
corruption (T-08-18 / T-08-19).

The module declares ``#[pymodule(gil_used = false)]`` in ``src/lib.rs``; this
test is the runtime evidence that the contract is sound.

Running it (PHASE GATE before ``/gsd-verify-work``)
---------------------------------------------------
This test is *meaningful only* on a free-threaded interpreter (``python3.13t``
or ``python3.14t``, PEP 703). On a standard GIL build it would exercise
serialized threads and could neither prove nor disprove the free-threaded
contract — so it **skips** (it does not pass, and it does not fail). This
mirrors the Phase-7.5 cpu-skip-guard lesson: a guard that *skips* on the wrong
runtime, rather than a false-pass or a panic.

To run it for real, on a free-threaded interpreter::

    # 1. obtain a free-threaded interpreter (build or install python3.13t)
    # 2. create a venv and install maturin into it
    python3.13t -m venv .venv-ft
    .venv-ft/bin/pip install maturin pytest numpy
    # 3. build the extension against the free-threaded interpreter
    cd crates/catboost-rs-py
    ../../.venv-ft/bin/maturin develop --features cpu
    # 4. run THIS test (it will NOT skip on a free-threaded build)
    ../../.venv-ft/bin/python -m pytest tests/test_free_threaded.py -q

On a GIL build the same command runs and reports the test as skipped.
"""

import sys
import threading

import numpy as np
import pytest


def _gil_enabled() -> bool:
    """True when the GIL is active (standard build, or any pre-3.13 build).

    ``sys._is_gil_enabled`` only exists on CPython >= 3.13; on older builds
    (e.g. the 3.12 abi3 test venv) the GIL is unconditionally enabled, so we
    treat the missing attribute as "GIL enabled" → this whole module skips.
    """
    is_gil_enabled = getattr(sys, "_is_gil_enabled", None)
    if is_gil_enabled is None:
        return True
    return bool(is_gil_enabled())


# The entire module is meaningful only under a free-threaded interpreter; skip
# (never panic, never false-pass) on a GIL build so the standard cpu CI run is a
# clean skip.
pytestmark = pytest.mark.skipif(
    _gil_enabled(),
    reason=(
        "free-threaded interpreter (python3.13t/3.14t) required for PYAPI-06 "
        "buffer-safety validation; sys._is_gil_enabled() is True (or absent on "
        "pre-3.13). See module docstring for the run command."
    ),
)

_N_THREADS = 8


def test_module_imports_free_threaded():
    """Under a free-threaded build the module imports (proves gil_used=false load).

    If ``catboost_rs`` were not declared ``gil_used=false``, importing it on a
    free-threaded interpreter would re-enable the GIL with a warning; the import
    succeeding here is the load-time half of the PYAPI-06 contract.
    """
    import catboost_rs

    assert hasattr(catboost_rs, "CatBoostRegressor")
    # On a free-threaded build, importing a gil_used=false module must NOT have
    # re-enabled the GIL.
    assert not _gil_enabled(), "importing catboost_rs re-enabled the GIL"


def test_concurrent_fit_predict_private_inputs(toy_regression):
    """N threads each fit a fresh estimator + predict on a private copy.

    Each worker owns its own copy of the input, so this exercises the
    own-before-detach path under genuine parallelism. All predictions must be
    finite, and — because every worker trains on an identical deterministic
    fixture — every worker must produce the *same* prediction vector
    (cross-thread equality), proving no cross-thread state corruption.
    """
    import catboost_rs

    x, y = toy_regression
    results: list = [None] * _N_THREADS
    errors: list = [None] * _N_THREADS

    def worker(idx: int) -> None:
        try:
            # Private per-thread copies (own-before-detach over distinct buffers).
            xi = np.ascontiguousarray(x.copy(), dtype=np.float32)
            yi = np.ascontiguousarray(y.copy(), dtype=np.float32)
            model = catboost_rs.CatBoostRegressor(iterations=10, depth=3)
            model.fit(xi, yi)
            results[idx] = model.predict(xi)
        except Exception as exc:  # noqa: BLE001 - record, re-raise in main thread
            errors[idx] = exc

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(_N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(e is None for e in errors), f"worker errors: {errors}"
    for idx, preds in enumerate(results):
        assert preds is not None, f"thread {idx} produced no result"
        assert preds.shape == (x.shape[0],)
        assert np.all(np.isfinite(preds)), f"thread {idx} produced non-finite preds"

    # Deterministic fixture + identical config => identical predictions per thread.
    baseline = results[0]
    for idx, preds in enumerate(results[1:], start=1):
        np.testing.assert_allclose(
            preds,
            baseline,
            rtol=0,
            atol=0,
            err_msg=f"thread {idx} diverged from thread 0 (corruption)",
        )


def test_concurrent_predict_shared_immutable_input(toy_regression):
    """N threads concurrently predict on ONE fitted model + ONE shared input.

    The fitted model (``Model`` is ``Send + Sync``, CLAUDE.md architecture) and
    the single immutable input array are shared read-only across all workers.
    ``predict`` takes ``&self`` over owned/quantized data, so concurrent reads
    must be race-free (T-08-19). Every worker must return the identical finite
    vector.
    """
    import catboost_rs

    x, y = toy_regression
    x_shared = np.ascontiguousarray(x.copy(), dtype=np.float32)
    x_shared.setflags(write=False)  # immutable: assert no thread mutates it

    model = catboost_rs.CatBoostRegressor(iterations=10, depth=3)
    model.fit(np.ascontiguousarray(x.copy(), dtype=np.float32),
              np.ascontiguousarray(y.copy(), dtype=np.float32))

    expected = model.predict(np.ascontiguousarray(x_shared, dtype=np.float32))

    results: list = [None] * _N_THREADS
    errors: list = [None] * _N_THREADS

    def worker(idx: int) -> None:
        try:
            # Share the single immutable buffer across all threads.
            results[idx] = model.predict(np.ascontiguousarray(x_shared, dtype=np.float32))
        except Exception as exc:  # noqa: BLE001
            errors[idx] = exc

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(_N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(e is None for e in errors), f"worker errors: {errors}"
    for idx, preds in enumerate(results):
        assert preds is not None, f"thread {idx} produced no result"
        assert np.all(np.isfinite(preds)), f"thread {idx} produced non-finite preds"
        np.testing.assert_allclose(
            preds,
            expected,
            rtol=0,
            atol=0,
            err_msg=f"thread {idx} diverged from single-threaded baseline (race)",
        )

    # The shared input must be untouched (own-before-detach copies before compute).
    assert not x_shared.flags.writeable
