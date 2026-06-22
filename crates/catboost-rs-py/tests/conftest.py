"""Shared pytest fixtures for the catboost_rs binding test suite.

Adds (08-04) a binary ``toy_classification`` fixture, a grouped ``toy_ranking``
fixture, and the offline catboost 1.2.10 oracle fixture paths under
``crates/cb-oracle/fixtures`` for the Python-surface parity test.
"""

import pathlib

import numpy as np
import pytest

# Repo root: this file is crates/catboost-rs-py/tests/conftest.py -> up 3.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_ORACLE_FIXTURES = _REPO_ROOT / "crates" / "cb-oracle" / "fixtures"


@pytest.fixture
def toy_regression():
    """A small C-contiguous float32 regression dataset.

    Returns ``(X, y)`` where ``X`` is a ``(50, 4)`` C-contiguous float32 matrix
    and ``y`` is a ``(50,)`` C-contiguous float32 target that is a deterministic
    linear function of the features plus a fixed perturbation (no RNG, so the
    fixture is reproducible).
    """
    rng = np.random.default_rng(0)
    x = np.ascontiguousarray(rng.standard_normal((50, 4)), dtype=np.float32)
    coef = np.array([1.5, -2.0, 0.5, 3.0], dtype=np.float32)
    y = np.ascontiguousarray(x @ coef + 0.1, dtype=np.float32)
    return x, y


@pytest.fixture
def toy_classification():
    """A small C-contiguous float32 BINARY classification dataset.

    Returns ``(X, y)`` where ``X`` is a ``(50, 4)`` C-contiguous float32 matrix
    and ``y`` is a ``(50,)`` C-contiguous float32 0/1 label thresholded at the
    median of a deterministic linear score (no RNG beyond the seeded features, so
    the fixture is reproducible). Both classes are present.
    """
    rng = np.random.default_rng(1)
    x = np.ascontiguousarray(rng.standard_normal((50, 4)), dtype=np.float32)
    coef = np.array([1.5, -2.0, 0.5, 3.0], dtype=np.float32)
    score = x @ coef
    y = np.ascontiguousarray((score > np.median(score)).astype(np.float32))
    return x, y


@pytest.fixture
def toy_ranking():
    """A small grouped ranking dataset.

    Returns ``(X, y, group_id)`` where ``X`` is a ``(60, 4)`` C-contiguous
    float32 matrix, ``y`` is a ``(60,)`` C-contiguous float32 relevance label,
    and ``group_id`` is a length-60 list of ``int`` group ids (10 groups of 6
    objects each). Suitable for constructing a ``Pool(..., group_id=...)``.
    """
    rng = np.random.default_rng(2)
    n_groups, per_group = 10, 6
    n = n_groups * per_group
    x = np.ascontiguousarray(rng.standard_normal((n, 4)), dtype=np.float32)
    coef = np.array([1.0, -1.0, 0.5, 2.0], dtype=np.float32)
    y = np.ascontiguousarray((x @ coef), dtype=np.float32)
    group_id = [g for g in range(n_groups) for _ in range(per_group)]
    return x, y, group_id


@pytest.fixture
def oracle_regression():
    """Offline catboost 1.2.10 regression oracle fixture (model_serde).

    Returns a dict with the reference ``.cbm`` model path, the input matrix
    (``numeric_tiny``), and the stored ``RawFormulaVal`` reference predictions.
    Hermetic: reads only frozen files, never imports the ``catboost`` package.
    """
    base = _ORACLE_FIXTURES / "model_serde" / "regression"
    x = np.load(_ORACLE_FIXTURES / "inputs" / "numeric_tiny" / "X.npy")
    ref = np.load(base / "predictions.npy")
    return {
        "cbm": str(base / "model.cbm"),
        "json": str(base / "model.json"),
        "X": np.ascontiguousarray(x, dtype=np.float32),
        "ref": ref,
    }


@pytest.fixture
def oracle_binclf():
    """Offline catboost 1.2.10 binary-classification oracle fixture (model_serde).

    Returns a dict with the reference ``.cbm`` model path, the input matrix
    (``numeric_tiny``), and the stored ``RawFormulaVal`` reference predictions
    (raw logits — convert with a logistic sigmoid for probabilities). Hermetic:
    reads only frozen files, never imports the ``catboost`` package.
    """
    base = _ORACLE_FIXTURES / "model_serde" / "binclf"
    x = np.load(_ORACLE_FIXTURES / "inputs" / "numeric_tiny" / "X.npy")
    ref_raw = np.load(base / "predictions.npy")
    return {
        "cbm": str(base / "model.cbm"),
        "json": str(base / "model.json"),
        "X": np.ascontiguousarray(x, dtype=np.float32),
        "ref_raw": ref_raw,
    }
