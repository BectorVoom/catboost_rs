"""PYAPI-05 Python-visible exception taxonomy tests.

The four exception types are importable from `catboost_rs`, the subclass
relationships hold (so `except CatBoostError` catches all of them, and sklearn's
not-fitted path recognizes `NotFittedError` as a `ValueError`), and a facade
`FeatureMismatch` surfaces as a catchable `CatBoostValueError`.
"""

import numpy as np
import pytest

import catboost_rs
from catboost_rs import (
    CatBoostError,
    CatBoostParameterError,
    CatBoostValueError,
    NotFittedError,
    CatBoostRegressor,
)


def test_exception_types_importable():
    for name in (
        "CatBoostError",
        "CatBoostParameterError",
        "CatBoostValueError",
        "NotFittedError",
    ):
        assert hasattr(catboost_rs, name), f"{name} not exported"
        assert isinstance(getattr(catboost_rs, name), type)


def test_subclass_relationships():
    # Every typed error is catchable as the base CatBoostError.
    assert issubclass(CatBoostParameterError, CatBoostError)
    assert issubclass(CatBoostValueError, CatBoostError)
    assert issubclass(NotFittedError, CatBoostError)
    # CatBoostError itself is an Exception.
    assert issubclass(CatBoostError, Exception)
    # NotFittedError must ALSO be a ValueError (sklearn check_is_fitted parity).
    assert issubclass(NotFittedError, ValueError)


def test_base_catches_subclasses():
    # `except CatBoostError` catches the parameter error.
    with pytest.raises(CatBoostError):
        raise CatBoostParameterError("x")
    with pytest.raises(CatBoostError):
        raise CatBoostValueError("y")
    with pytest.raises(CatBoostError):
        raise NotFittedError("z")


def test_not_fitted_catchable_as_value_error():
    # sklearn's not-fitted path catches ValueError; NotFittedError must qualify.
    with pytest.raises(ValueError):
        raise NotFittedError("not fitted")


def test_not_fitted_raised_on_predict_before_fit():
    model = CatBoostRegressor(iterations=2, depth=2)
    x = np.zeros((3, 2), dtype=np.float32)
    with pytest.raises(NotFittedError):
        model.predict(x)


def test_feature_mismatch_surfaces_as_value_error():
    # Fit on k=2 features, predict on k=3 -> facade FeatureMismatch ->
    # CatBoostValueError, message mentions the mismatch.
    x_train = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]], dtype=np.float32)
    y_train = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    model = CatBoostRegressor(iterations=3, depth=2)
    model.fit(x_train, y_train)

    x_bad = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
    with pytest.raises(CatBoostValueError) as excinfo:
        model.predict(x_bad)
    # Also catchable as the base.
    assert issubclass(type(excinfo.value), CatBoostError)
