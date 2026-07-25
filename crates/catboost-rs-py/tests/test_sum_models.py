"""SM-06 Python-surface test: module-level `catboost_rs.sum_models`.

Mirrors upstream `catboost.sum_models(models, weights=None)`: combines several
fitted `CatBoostRegressor`s into one whose `.predict(X)` equals the weighted
sum of the inputs' predictions.
"""

import numpy as np


def test_sum_models_predict_equals_weighted_sum(toy_regression):
    """Two fitted regressors summed with default (all-ones) weights predict
    the elementwise sum of their individual predictions (<=1e-5)."""
    import catboost_rs

    x, y = toy_regression
    a = catboost_rs.CatBoostRegressor(iterations=10, depth=3, random_seed=1).fit(x, y)
    b = catboost_rs.CatBoostRegressor(iterations=8, depth=2, random_seed=2).fit(x, y)

    merged = catboost_rs.sum_models([a, b])
    expected = a.predict(x) + b.predict(x)

    np.testing.assert_allclose(merged.predict(x), expected, atol=1e-5)


def test_sum_models_applies_weights(toy_regression):
    """Explicit weights scale each model's contribution before summing."""
    import catboost_rs

    x, y = toy_regression
    a = catboost_rs.CatBoostRegressor(iterations=10, depth=3, random_seed=1).fit(x, y)
    b = catboost_rs.CatBoostRegressor(iterations=8, depth=2, random_seed=2).fit(x, y)

    merged = catboost_rs.sum_models([a, b], weights=[0.25, 2.0])
    expected = 0.25 * a.predict(x) + 2.0 * b.predict(x)

    np.testing.assert_allclose(merged.predict(x), expected, atol=1e-5)


def test_sum_models_unfitted_input_raises_not_fitted(toy_regression):
    """An unfitted model in the input list raises NotFittedError, not a panic."""
    import catboost_rs

    x, y = toy_regression
    a = catboost_rs.CatBoostRegressor(iterations=10, depth=3).fit(x, y)
    unfitted = catboost_rs.CatBoostRegressor(iterations=10, depth=3)

    raised = False
    try:
        catboost_rs.sum_models([a, unfitted])
    except catboost_rs.NotFittedError:
        raised = True
    assert raised


def test_sum_models_weight_count_mismatch_raises(toy_regression):
    """A weights list whose length disagrees with the model count is a typed
    ValueError-family error, not a panic."""
    import catboost_rs

    x, y = toy_regression
    a = catboost_rs.CatBoostRegressor(iterations=10, depth=3).fit(x, y)
    b = catboost_rs.CatBoostRegressor(iterations=10, depth=3).fit(x, y)

    raised = False
    try:
        catboost_rs.sum_models([a, b], weights=[1.0])
    except catboost_rs.CatBoostError:
        raised = True
    assert raised
