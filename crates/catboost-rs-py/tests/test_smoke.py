"""08-01 walking-skeleton smoke test: import + fit + predict end-to-end.

Proves the full stack travels the whole boundary (NumPy borrow -> own ->
OwnedColumns::into_pool -> CatBoostBuilder::fit -> Model::predict -> NumPy out)
through the real catboost-rs facade — not a stub.
"""

import numpy as np


def test_import():
    """`import catboost_rs` succeeds and exposes CatBoostRegressor."""
    import catboost_rs

    assert hasattr(catboost_rs, "CatBoostRegressor")


def test_fit_predict_end_to_end(toy_regression):
    """fit(X32, y32).predict(X32) returns a finite float array of length n."""
    import catboost_rs

    x, y = toy_regression
    model = catboost_rs.CatBoostRegressor(iterations=10, depth=3)
    fitted = model.fit(x, y)
    # fit returns the estimator (sklearn convention).
    preds = fitted.predict(x)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (x.shape[0],)
    assert np.all(np.isfinite(preds))


def test_predictions_deterministic(toy_regression):
    """Two fits with the same seed produce identical predictions (sanity)."""
    import catboost_rs

    x, y = toy_regression

    m1 = catboost_rs.CatBoostRegressor(iterations=10, depth=3, random_seed=42)
    p1 = m1.fit(x, y).predict(x)

    m2 = catboost_rs.CatBoostRegressor(iterations=10, depth=3, random_seed=42)
    p2 = m2.fit(x, y).predict(x)

    np.testing.assert_array_equal(p1, p2)


def test_predict_before_fit_raises(toy_regression):
    """predict before fit raises (placeholder for the typed NotFittedError)."""
    import catboost_rs

    x, _ = toy_regression
    model = catboost_rs.CatBoostRegressor(iterations=5)
    raised = False
    try:
        model.predict(x)
    except ValueError:
        raised = True
    assert raised


def test_float64_rejected(toy_regression):
    """A float64 X is rejected (D-12: no silent precision coercion).

    Since 08-03 the ingest path raises the typed ``CatBoostValueError`` (a
    subclass of ``CatBoostError``) instead of the bare stdlib ``ValueError``.
    """
    import catboost_rs

    x, y = toy_regression
    model = catboost_rs.CatBoostRegressor(iterations=5)
    raised = False
    try:
        model.fit(x.astype(np.float64), y)
    except catboost_rs.CatBoostValueError:
        raised = True
    assert raised
