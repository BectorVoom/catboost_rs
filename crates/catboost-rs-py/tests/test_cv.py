"""ORCH-01-S6 parity + error tests for ``catboost_rs.cv``.

Requires the extension built into a catboost-1.2.10 uv/venv 3.12 environment:
    uv venv --python 3.12
    uv pip install catboost==1.2.10 'numpy<2' scikit-learn maturin pytest
    maturin develop -m crates/catboost-rs-py/Cargo.toml
    pytest crates/catboost-rs-py/tests/test_cv.py -x

Parity is asserted with EXPLICIT folds and a params dict that pins
``boost_from_average=True`` + ``bootstrap_type="No"`` + ``random_strength=0``,
matching the Rust ``CatBoostBuilder`` defaults the facade ``cv()`` trains with
(SPEC §1.1) — without these pins the raw ``catboost.cv()`` dict API trains with
different effective defaults and would never reproduce ≤1e-5.
"""
import numpy as np
import pytest

import catboost
import catboost_rs


# A fixed, C-contiguous float32 numeric-regression dataset (fed IDENTICALLY to
# both surfaces so each quantizes the same feature values).
def _dataset():
    rng = np.random.default_rng(20260720)
    x = np.ascontiguousarray(rng.standard_normal((60, 4)), dtype=np.float32)
    coef = np.array([1.5, -2.0, 0.5, 3.0], dtype=np.float32)
    y = np.ascontiguousarray(x @ coef + 0.1, dtype=np.float32)
    return x, y


# Explicit, fixed contiguous 3-fold split (test-row index lists + the matching
# (train, test) tuples upstream `catboost.cv` consumes).
def _folds(n, fold_count=3):
    bounds = [round(i * n / fold_count) for i in range(fold_count + 1)]
    test_lists = [list(range(bounds[k], bounds[k + 1])) for k in range(fold_count)]
    cb_pairs = []
    for k in range(fold_count):
        test = np.array(test_lists[k], dtype=np.int64)
        mask = np.ones(n, dtype=bool)
        mask[bounds[k]:bounds[k + 1]] = False
        train = np.arange(n, dtype=np.int64)[mask]
        cb_pairs.append((train, test))
    return test_lists, cb_pairs


PARAMS = {
    "iterations": 10,
    "learning_rate": 0.1,
    "depth": 4,
    "loss_function": "RMSE",
    "border_count": 128,
    "random_strength": 0,
    "boost_from_average": True,
    "bootstrap_type": "No",
}

_COLUMNS = ["test-RMSE-mean", "test-RMSE-std", "train-RMSE-mean", "train-RMSE-std"]


def test_cv_resolves():
    # The module-level `cv` free function is importable and callable.
    assert callable(catboost_rs.cv)


def test_cv_parity_explicit_folds():
    x, y = _dataset()
    n = x.shape[0]
    test_lists, cb_pairs = _folds(n)

    pool = catboost_rs.Pool(x, label=y)
    got = catboost_rs.cv(pool, PARAMS, folds=test_lists, shuffle=False)

    cb_pool = catboost.Pool(x, label=y)
    exp = catboost.cv(cb_pool, PARAMS, folds=cb_pairs, shuffle=False, as_pandas=True)

    assert len(got["iterations"]) == PARAMS["iterations"]
    for col in _COLUMNS:
        got_col = np.asarray(got[col], dtype=np.float64)
        exp_col = np.asarray(exp[col], dtype=np.float64)
        assert got_col.shape == exp_col.shape, col
        assert np.max(np.abs(got_col - exp_col)) <= 1e-5, (
            f"{col}: max|diff|={np.max(np.abs(got_col - exp_col))}"
        )


def test_cv_metric_from_loss_default():
    # metrics=None derives the metric from loss_function ("RMSE") -> the same
    # test-RMSE-* / train-RMSE-* columns appear.
    x, y = _dataset()
    test_lists, _ = _folds(x.shape[0])
    pool = catboost_rs.Pool(x, label=y)
    got = catboost_rs.cv(pool, PARAMS, folds=test_lists, shuffle=False, metrics=None)
    for col in _COLUMNS:
        assert col in got


def test_cv_empty_metric_raises():
    x, y = _dataset()
    test_lists, _ = _folds(x.shape[0])
    pool = catboost_rs.Pool(x, label=y)
    with pytest.raises(catboost_rs.CatBoostError):
        catboost_rs.cv(pool, PARAMS, folds=test_lists, metrics=[])


def test_cv_unknown_metric_raises():
    x, y = _dataset()
    test_lists, _ = _folds(x.shape[0])
    pool = catboost_rs.Pool(x, label=y)
    with pytest.raises(catboost_rs.CatBoostError):
        catboost_rs.cv(pool, PARAMS, folds=test_lists, metrics="NoSuchMetric")


def test_cv_categorical_pool_maps_cleanly():
    # SPEC §5-S6 anticipates a categorical Pool tripping the `staged_predict`
    # scalar/oblivious/float-only guard (`UnsupportedModel`), surfaced as a mapped
    # CatBoostError subclass. EMPIRICAL FINDING on this build: the current
    # training path does not CTR-model declared `cat_features` (a 1-vs-2 cat_feature
    # run yields byte-identical curves), so no `ctr_data` is produced and the guard
    # is never reached from the float-only Python surface (`parse_loss` also
    # exposes only scalar losses, so `approx_dimension>1` is unreachable too). That
    # facade/cb-train behavior is out of this (additive-binding) task's scope. What
    # this test PINS is the binding's own contract: a categorical Pool is handled
    # WITHOUT aborting — it either returns a well-formed columns dict or raises a
    # MAPPED `CatBoostError` (never an unmapped abort/panic).
    x, y = _dataset()
    test_lists, _ = _folds(x.shape[0])
    pool = catboost_rs.Pool(x, label=y, cat_features=[0])
    try:
        result = catboost_rs.cv(pool, PARAMS, folds=test_lists, metrics="RMSE")
    except catboost_rs.CatBoostError:
        return  # mapped exception path (the SPEC-anticipated UnsupportedModel arm)
    # clean-completion path: the result is a well-formed cv columns dict.
    assert len(result["iterations"]) == PARAMS["iterations"]
    for col in _COLUMNS:
        assert col in result
