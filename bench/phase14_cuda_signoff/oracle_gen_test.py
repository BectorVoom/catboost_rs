#!/usr/bin/env python3
"""Offline test of the importable gen() workload in oracle.py (no GPU, no Kaggle run).

Proves the numpy gen() reproduces crates/cb-train/tests/bench_grow_speed_test.rs::gen()
(lines 43-65): shape (n x 20) float32, integer bins in [0, 32), determinism across
calls, +/-1 target with both classes present, and spot-checked cells matching the
hash formula (i*2654435761 + f*40503) % 32. 32 divides 2^64, so the mod-2^64
wrapping used in Rust does not affect the final bin — the plain formula is exact.

Kept separate from oracle.py (CLAUDE.md source/test separation). Importing the
sibling module via importlib does NOT run main() (guarded by __main__).
Skips cleanly if numpy is unavailable rather than failing the suite.
"""
import importlib.util
import os

import pytest

np = pytest.importorskip("numpy")


def _load_oracle():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "oracle.py")
    spec = importlib.util.spec_from_file_location("oracle14_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # safe: run body guarded by if __name__ == "__main__"
    return module


ORACLE = _load_oracle()


def test_shape_and_dtype():
    X, y = ORACLE.gen(1000)
    assert X.shape == (1000, 20)
    assert X.dtype == np.float32
    assert y.shape == (1000,)


def test_integer_bins_in_range():
    X, _ = ORACLE.gen(1000)
    # every value is an integer bin in [0, 32)
    assert X.min() >= 0.0
    assert X.max() < 32.0
    assert np.all(X == np.floor(X)), "bins must be integer-valued float32"


def test_determinism():
    X1, y1 = ORACLE.gen(1000)
    X2, y2 = ORACLE.gen(1000)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)


def test_target_is_pm1_both_classes():
    _, y = ORACLE.gen(5000)
    assert set(y.tolist()) <= {1.0, -1.0}
    assert 1.0 in set(y.tolist())
    assert -1.0 in set(y.tolist())


def test_spot_check_hash_formula():
    X, _ = ORACLE.gen(64)
    # 32 divides 2^64, so wrapping mod 2^64 then mod 32 == plain mod 32.
    for i in (0, 1, 7, 31, 63):
        for f in (0, 1, 5, 19):
            expected = (i * 2654435761 + f * 40503) % 32
            assert float(X[i, f]) == float(expected), (i, f, X[i, f], expected)


def test_target_matches_source_rule():
    X, y = ORACLE.gen(2000)
    a = X[:, 0].astype(np.float64)
    b = X[:, 1].astype(np.float64)  # 1 % 20 == 1
    expected = np.where(a + 0.5 * b > 32.0 * 0.75, 1.0, -1.0)
    assert np.array_equal(y, expected)
