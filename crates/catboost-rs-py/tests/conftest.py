"""Shared pytest fixtures for the catboost_rs binding smoke suite (08-01)."""

import numpy as np
import pytest


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
