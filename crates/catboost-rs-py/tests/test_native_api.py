"""Native-API tests for the CatBoost-mirror estimator trio (08-04, PYAPI-03).

Covers the classifier (`fit` / `predict` class labels / `predict_proba` `(n, 2)`)
and the ranker (`fit` over a grouped `Pool` / `predict` scores; group_id-absence
rejection). The shape conventions asserted here are the contract the oracle-parity
test (`test_oracle_parity.py`) then pins numerically against catboost 1.2.10.
"""

import numpy as np
import pytest

import catboost_rs


# --- CatBoostClassifier -----------------------------------------------------


def test_classifier_predict_returns_class_labels(toy_classification):
    """`predict` returns class labels of shape (n,) drawn from {0, 1}."""
    x, y = toy_classification
    clf = catboost_rs.CatBoostClassifier(iterations=20, depth=3).fit(x, y)
    labels = clf.predict(x)
    assert labels.shape == (x.shape[0],)
    assert set(np.unique(labels)).issubset({0.0, 1.0})


def test_classifier_predict_proba_shape_is_n_by_2(toy_classification):
    """`predict_proba` returns probabilities of shape (n, 2), rows summing to 1.

    The chosen convention is the upstream binary `(n, 2)` form
    (`[P(class 0), P(class 1)]` per row).
    """
    x, y = toy_classification
    clf = catboost_rs.CatBoostClassifier(iterations=20, depth=3).fit(x, y)
    proba = clf.predict_proba(x)
    assert proba.shape == (x.shape[0], 2)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_classifier_defaults_to_classification_loss(toy_classification):
    """No `loss_function` passed => a classification model (proba in [0,1])."""
    x, y = toy_classification
    # No loss_function kwarg: the classifier must default to Logloss, not RMSE.
    clf = catboost_rs.CatBoostClassifier(iterations=20).fit(x, y)
    proba = clf.predict_proba(x)
    # A regression default (RMSE) cannot produce a valid 2-column probability
    # simplex; this both-classes-present, rows-sum-to-1 check pins the default.
    assert proba.shape == (x.shape[0], 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# --- CatBoostRanker ---------------------------------------------------------


def test_ranker_fit_predict_over_grouped_pool(toy_ranking):
    """`fit` on a group_id Pool then `predict` returns finite scores (n,)."""
    x, y, group_id = toy_ranking
    pool = catboost_rs.Pool(x, label=y, group_id=group_id)
    ranker = catboost_rs.CatBoostRanker(iterations=20).fit(pool)
    scores = ranker.predict(x)
    assert scores.shape == (x.shape[0],)
    assert np.all(np.isfinite(scores))


def test_ranker_without_group_id_raises_actionable(toy_ranking):
    """Fitting a ranker on a group-less dataset raises a typed, actionable error."""
    x, y, _ = toy_ranking
    ranker = catboost_rs.CatBoostRanker(iterations=20)
    with pytest.raises(catboost_rs.CatBoostValueError) as exc:
        ranker.fit(x, y)
    assert "group_id" in str(exc.value)


def test_ranker_pool_without_group_id_raises(toy_ranking):
    """A native Pool built WITHOUT group_id is also rejected by the ranker."""
    x, y, _ = toy_ranking
    pool = catboost_rs.Pool(x, label=y)  # no group_id
    ranker = catboost_rs.CatBoostRanker(iterations=20)
    with pytest.raises(catboost_rs.CatBoostValueError):
        ranker.fit(pool)
