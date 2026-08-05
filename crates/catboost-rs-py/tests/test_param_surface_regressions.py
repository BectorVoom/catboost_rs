"""Regression locks for the Python param/ingestion surface.

Three defects the fit-param promotion (PARAM-01..03) made reachable:

1. **Scoring a native ``Pool`` after a categorical fit was impossible.**
   ``predict`` / ``predict_proba`` / ``score`` / ``partial_dependence`` pass the
   estimator's REMEMBERED fit-time ``cat_features`` down to the ingest helper,
   which raises whenever a non-empty ``cat_features`` accompanies a ``Pool``.
   The caller had passed no ``cat_features`` at all.

2. **The Python literal ``None`` crashed the newly promoted params.**
   ``None`` is upstream's universal "not set" and is the declared default of
   ``auto_class_weights`` / ``od_type`` / ``eval_metric`` /
   ``monotone_constraints``. Extracting it as ``str`` raised a PyO3 ``TypeError``
   before any of this crate's own parsing could run — breaking explicit
   ``param=None`` defaults and every ``sklearn.clone`` / ``get_params``
   round-trip.

3. **Validation was asymmetric across entry points.** ``fit`` rejected the
   eval-set-only params (by NAME, even at their own disabling values), while
   ``cv`` / ``grid_search`` / ``randomized_search`` accepted and silently
   dropped them.
"""

import warnings

import numpy as np
import pytest

import catboost_rs
from catboost_rs import (
    CatBoostClassifier,
    CatBoostParameterError,
    CatBoostRegressor,
    Pool,
)


def _cat_frame(n=40, seed=0):
    """A 2-float + 1-categorical dataset as the trailing-cat-block layout the
    ingest path requires, plus a binary label."""
    rng = np.random.default_rng(seed)
    f0 = rng.random(n)
    f1 = rng.random(n)
    cat = np.array(["alpha" if i % 2 == 0 else "beta" for i in range(n)], dtype=object)
    # `Pool` requires a 1-D float32 label.
    y = np.ascontiguousarray((f0 > 0.5).astype(np.float32))
    return f0, f1, cat, y


def _cat_pool(n=40, seed=0):
    """Build a ``Pool`` with column 2 categorical (a trailing block)."""
    pd = pytest.importorskip("pandas")
    f0, f1, cat, y = _cat_frame(n, seed)
    df = pd.DataFrame({"f0": f0, "f1": f1, "c": cat})
    return Pool(df, y, cat_features=[2]), df, y


# ---------------------------------------------------------------------------
# 1. predict(Pool) after a categorical fit
# ---------------------------------------------------------------------------


def test_predict_accepts_a_pool_after_a_categorical_fit():
    """``fit(df, y, cat_features=[2])`` then ``predict(Pool(...))`` must work.

    The ``cat_features`` in play is the estimator's own record of the fit, not
    an argument the caller passed, so the "cat_features cannot be given when the
    data is a Pool" ambiguity rule must not fire.
    """
    pool, df, y = _cat_pool()
    clf = CatBoostClassifier(iterations=5, depth=3, one_hot_max_size=2)
    clf.fit(df, y, cat_features=[2])

    preds = clf.predict(pool)
    assert len(preds) == len(y)

    # The same call through the other scoring entry points.
    proba = clf.predict_proba(pool)
    assert proba.shape[0] == len(y)
    assert clf.score(pool, y) is not None


def test_predict_still_rejects_an_explicit_cat_features_with_a_pool():
    """The ambiguity rule itself is intact for an EXPLICIT argument."""
    pool, df, y = _cat_pool()
    from catboost_rs import CatBoostValueError

    with pytest.raises(CatBoostValueError, match="cat_features cannot be given"):
        CatBoostClassifier(iterations=3).fit(pool, y, cat_features=[2])


# ---------------------------------------------------------------------------
# 2. `None` means "not set"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "auto_class_weights",
        "od_type",
        "eval_metric",
        "monotone_constraints",
        "early_stopping_rounds",
        "class_weights",
        "scale_pos_weight",
        "ignored_features",
        "grow_policy",
    ],
)
def test_none_valued_params_are_treated_as_unset(name):
    rng = np.random.default_rng(0)
    x = rng.random((40, 3)).astype(np.float32)
    y = (x[:, 0] > 0.5).astype(np.float32)
    clf = CatBoostClassifier(iterations=5, depth=2, **{name: None})
    clf.fit(x, y)  # must not raise TypeError
    assert len(clf.predict(x)) == len(y)


def test_sklearn_clone_round_trip_refits():
    """``sklearn.clone`` reconstructs via ``__init__(**get_params())``.

    A wrapper that materializes ``param=None`` defaults produces exactly the
    dict this exercises.
    """
    sklearn_base = pytest.importorskip("sklearn.base")
    rng = np.random.default_rng(1)
    x = rng.random((40, 3)).astype(np.float32)
    y = (x[:, 0] > 0.5).astype(np.float32)

    est = CatBoostClassifier(
        iterations=5,
        depth=2,
        auto_class_weights=None,
        od_type=None,
        eval_metric=None,
        monotone_constraints=None,
    )
    clone = sklearn_base.clone(est)
    clone.fit(x, y)
    assert len(clone.predict(x)) == len(y)


def test_a_none_valued_grid_point_is_accepted():
    """``{"auto_class_weights": [None, "Balanced"]}`` must be a legal sweep."""
    rng = np.random.default_rng(2)
    x = rng.random((60, 3)).astype(np.float32)
    y = (x[:, 0] > 0.5).astype(np.float32)
    # `loss_function` is pinned explicitly: `grid_search` reads the estimator's
    # `get_params()`, which holds only the kwargs actually passed, so the
    # classifier's implicit Logloss default would not reach the builder and the
    # class-weight candidate would be scored against a regression loss.
    est = CatBoostClassifier(iterations=5, depth=2, loss_function="Logloss")
    # Every candidate must have RUN. A `None` grid value that raised would be
    # caught by the facade's failure isolation and reported as a
    # `FitFailedWarning` + `error_score`, so the search would still "succeed"
    # while having trained nothing for that point — assert on the warning, not
    # just on the return.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = catboost_rs.grid_search(
            est,
            {"auto_class_weights": [None, "Balanced"]},
            x,
            y,
            cv=2,
            refit=False,
        )
    assert "params" in res
    failures = [w for w in caught if w.category is catboost_rs.FitFailedWarning]
    assert not failures, f"a grid candidate failed to fit: {[str(w.message) for w in failures]}"


# ---------------------------------------------------------------------------
# 3. Validation symmetry across fit / cv / grid_search
# ---------------------------------------------------------------------------


def _xy(n=60, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.random((n, 3)).astype(np.float32)
    y = (x[:, 0] * 2.0 - x[:, 1]).astype(np.float32)
    return x, y


@pytest.mark.parametrize(
    "param",
    [
        {"early_stopping_rounds": 5},
        {"use_best_model": True},
        {"od_type": "Iter", "od_wait": 5},
    ],
)
def test_cv_rejects_eval_set_only_params(param):
    """``cv`` fits every fold through the learn-only path.

    Before this, ``fit`` raised on these while ``cv`` accepted them and silently
    ran every fold to completion — and the raise on ``fit`` is exactly what
    persuades a user the parameter is honoured on ``cv`` too.
    """
    x, y = _xy()
    with pytest.raises(CatBoostParameterError, match="validation"):
        catboost_rs.cv((x, y), {"iterations": 20, **param}, fold_count=2)


@pytest.mark.parametrize(
    "grid",
    [
        {"early_stopping_rounds": [5, 10]},
        {"use_best_model": [True]},
    ],
)
def test_grid_search_rejects_eval_set_only_params_in_the_grid(grid):
    x, y = _xy()
    est = CatBoostRegressor(iterations=20, depth=2)
    with pytest.raises(CatBoostParameterError, match="validation"):
        catboost_rs.grid_search(est, grid, x, y, cv=2, refit=False)


def test_grid_search_rejects_an_eval_set_only_param_inherited_from_the_estimator():
    x, y = _xy()
    est = CatBoostRegressor(iterations=20, depth=2, use_best_model=True)
    with pytest.raises(CatBoostParameterError, match="validation"):
        catboost_rs.grid_search(est, {"depth": [2, 3]}, x, y, cv=2, refit=False)


def test_randomized_search_rejects_eval_set_only_params():
    x, y = _xy()
    est = CatBoostRegressor(iterations=20, depth=2)
    with pytest.raises(CatBoostParameterError, match="validation"):
        catboost_rs.randomized_search(
            est, {"early_stopping_rounds": [5, 10]}, x, y, cv=2, n_iter=2, refit=False
        )


# ---------------------------------------------------------------------------
# 3b. ...and the guard keys on the VALUE, not the name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "param",
    [
        {"od_type": "None"},
        {"od_type": None},
        {"od_pval": 0.0},
        {"use_best_model": False},
        {"early_stopping_rounds": None},
        {"od_wait": None},
    ],
)
def test_fit_accepts_inert_eval_set_only_values_without_an_eval_set(param):
    """Explicitly DISABLING early stopping is not a request the learn-only path
    cannot serve, so it must not be rejected."""
    x, y = _xy()
    CatBoostRegressor(iterations=5, depth=2, **param).fit(x, y)


def test_fit_accepts_a_materialized_default_parameter_dict():
    """The shape a config layer / ``get_all_params`` round-trip produces."""
    x, y = _xy()
    CatBoostRegressor(
        iterations=5,
        depth=2,
        od_type="None",
        od_pval=0.0,
        od_wait=None,
        early_stopping_rounds=None,
        use_best_model=False,
    ).fit(x, y)


def test_fit_still_rejects_an_active_eval_set_only_param():
    """The guard is not defanged: an ACTIVE value with no eval_set still raises."""
    x, y = _xy()
    with pytest.raises(CatBoostParameterError, match="validation"):
        CatBoostRegressor(iterations=20, depth=2, early_stopping_rounds=5).fit(x, y)


def test_cv_accepts_inert_eval_set_only_values():
    x, y = _xy()
    res = catboost_rs.cv(
        (x, y),
        {"iterations": 5, "depth": 2, "use_best_model": False, "od_type": "None"},
        fold_count=2,
    )
    assert "iterations" in res
