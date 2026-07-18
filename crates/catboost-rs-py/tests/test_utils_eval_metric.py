"""ORCH-04-S6 parity test for ``catboost_rs.utils.eval_metric``.

Requires the extension built into a catboost-1.2.10 uv 3.12 venv:
    uv venv --python 3.12
    uv pip install catboost==1.2.10 'numpy<2' maturin pytest
    maturin develop
    pytest crates/catboost-rs-py/tests/test_utils_eval_metric.py
"""
import numpy as np
import pytest

import catboost_rs
import catboost_rs.utils  # submodule import form must work
from catboost_rs.utils import eval_metric  # from-import form must work
from catboost.utils import eval_metric as cb_eval_metric


# label in {0,1}; approx > -1 so RMSE + Logloss + MSLE are all valid.
LABEL = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0], dtype=np.float64)
APPROX = np.array([0.3, -0.4, 0.9, 0.1, 0.5, -0.2], dtype=np.float64)


def test_submodule_import_forms():
    # Both the attribute form and the from-import form resolve to a callable.
    assert callable(catboost_rs.utils.eval_metric)
    assert callable(eval_metric)


@pytest.mark.parametrize("metric", ["RMSE", "MSLE", "Logloss"])
def test_scalar_parity(metric):
    got = catboost_rs.utils.eval_metric(LABEL, APPROX, metric)
    assert isinstance(got, float)
    exp = np.asarray(cb_eval_metric(LABEL, APPROX, metric), dtype=np.float64).reshape(-1)[0]
    assert abs(got - exp) <= 1e-5, f"{metric}: got {got}, upstream {exp}"


def test_list_parity():
    got = catboost_rs.utils.eval_metric(LABEL, APPROX, ["RMSE", "MSLE"])
    assert isinstance(got, list)
    assert len(got) == 2
    for name, val in zip(["RMSE", "MSLE"], got):
        exp = np.asarray(cb_eval_metric(LABEL, APPROX, name), dtype=np.float64).reshape(-1)[0]
        assert abs(val - exp) <= 1e-5, f"{name}: got {val}, upstream {exp}"


def test_bad_metric_raises():
    with pytest.raises(catboost_rs.CatBoostError):
        catboost_rs.utils.eval_metric(LABEL, APPROX, "NoSuchMetric")


def test_length_mismatch_raises_value_error():
    # A `label`/`approx` length mismatch is a malformed-input value error, not a
    # base CatBoostError (so `except CatBoostValueError` catches it).
    with pytest.raises(catboost_rs.CatBoostValueError):
        catboost_rs.utils.eval_metric(LABEL, APPROX[:-1], "RMSE")


def test_explicit_empty_weight_raises_value_error():
    # An explicitly-supplied 0-length weight is a length mismatch (matching
    # upstream) — NOT silently coerced to uniform weights.
    with pytest.raises(catboost_rs.CatBoostValueError):
        catboost_rs.utils.eval_metric(LABEL, APPROX, "RMSE", weight=np.array([], dtype=np.float64))


def test_wrong_length_weight_raises_value_error():
    with pytest.raises(catboost_rs.CatBoostValueError):
        catboost_rs.utils.eval_metric(LABEL, APPROX, "RMSE", weight=np.ones(3, dtype=np.float64))


def test_negative_group_id_raises_value_error():
    # A negative (float) group id must error, never silently saturate to 0 and
    # merge distinct query groups.
    bad_group = np.array([-1.0, -1.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    with pytest.raises(catboost_rs.CatBoostValueError):
        catboost_rs.utils.eval_metric(LABEL, APPROX, "NDCG", group_id=bad_group)


def test_integer_group_id_parity():
    # Integer group ids extract losslessly and match upstream NDCG grouping.
    group = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    got = catboost_rs.utils.eval_metric(LABEL, APPROX, "NDCG", group_id=group)
    exp = np.asarray(
        cb_eval_metric(LABEL, APPROX, "NDCG", group_id=group), dtype=np.float64
    ).reshape(-1)[0]
    assert abs(got - exp) <= 1e-5, f"NDCG grouped: got {got}, upstream {exp}"
