"""ORCH-02-S5 — `catboost_rs.grid_search` parity / structure / error tests.

Mirrors the supported subset of upstream `CatBoost.grid_search(param_grid, X, y,
cv=3, ...)`: expand a `dict[str, list]` grid into the Cartesian product of
candidate param dicts, cross-validate each via the shipped `catboost_rs::cv`
seam, and return `{"params": <best grid point>, "cv_results": {columns}}`.

The primary oracle is DECOMPOSITION self-consistency: the returned
`cv_results["test-RMSE-mean"]` for the winning grid point must equal a direct
`catboost_rs.cv(<estimator with best params>, X, y, folds=<fixed>,
shuffle=False)` call to <=1e-5 (same underlying seams).
"""

import numpy as np
import pytest

import catboost_rs


def _toy_regression(n=48, k=4, seed=7):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, k)).astype(np.float32)
    # A smooth-ish target so shallow/deep trees actually differ in cv score.
    y = (x[:, 0] * 2.0 - x[:, 1] + 0.5 * x[:, 2] ** 2).astype(np.float32)
    return x, y


def _fixed_folds(n=48, k=3):
    """A deterministic contiguous k-fold assignment: fold f holds rows
    [f*n//k, (f+1)*n//k)."""
    folds = []
    for f in range(k):
        lo = f * n // k
        hi = (f + 1) * n // k
        folds.append(list(range(lo, hi)))
    return folds


def test_grid_search_symbol_resolves():
    assert hasattr(catboost_rs, "grid_search")
    assert callable(catboost_rs.grid_search)


def test_grid_search_structure_and_params_membership():
    x, y = _toy_regression()
    folds = _fixed_folds()
    reg = catboost_rs.CatBoostRegressor(
        loss_function="RMSE", iterations=15, learning_rate=0.2, random_seed=0
    )
    grid = {"depth": [3, 5], "learning_rate": [0.1, 0.3]}
    res = catboost_rs.grid_search(
        reg, grid, x, y, cv=3, folds=folds, shuffle=False, refit=False
    )
    assert set(res.keys()) == {"params", "cv_results"}
    cvr = res["cv_results"]
    assert "test-RMSE-mean" in cvr
    assert "iterations" in cvr
    # cv_results columns are per-iteration; length == iterations count.
    assert len(cvr["test-RMSE-mean"]) == len(cvr["iterations"])
    # "params" is one of the 4 grid points.
    grid_points = [
        {"depth": d, "learning_rate": lr} for d in [3, 5] for lr in [0.1, 0.3]
    ]
    assert res["params"] in grid_points


def test_grid_search_cv_self_consistency():
    x, y = _toy_regression()
    folds = _fixed_folds()
    base_kwargs = dict(
        loss_function="RMSE", iterations=15, learning_rate=0.2, random_seed=0
    )
    reg = catboost_rs.CatBoostRegressor(**base_kwargs)
    grid = {"depth": [3, 5], "learning_rate": [0.1, 0.3]}
    res = catboost_rs.grid_search(
        reg, grid, x, y, cv=3, folds=folds, shuffle=False, refit=False
    )
    best = dict(base_kwargs)
    best.update(res["params"])
    direct = catboost_rs.cv(
        (x, y), best, fold_count=3, folds=folds, shuffle=False
    )
    got = np.asarray(res["cv_results"]["test-RMSE-mean"], dtype=np.float64)
    exp = np.asarray(direct["test-RMSE-mean"], dtype=np.float64)
    assert got.shape == exp.shape
    assert np.max(np.abs(got - exp)) <= 1e-5


def test_grid_search_bad_param_raises():
    x, y = _toy_regression()
    folds = _fixed_folds()
    reg = catboost_rs.CatBoostRegressor(loss_function="RMSE", iterations=10)
    # `not_a_real_param` is unknown -> validate_params rejects it before training.
    with pytest.raises(catboost_rs.CatBoostError):
        catboost_rs.grid_search(
            reg,
            {"not_a_real_param": [1, 2]},
            x,
            y,
            cv=3,
            folds=folds,
            shuffle=False,
            refit=False,
        )


# The degenerate grid point: a Bayesian-bootstrap `bagging_temperature` so
# extreme that quantization finds no candidate split, so `cv` ERRORS for that
# candidate (empirically verified — the Python param validator rejects the
# simpler `iterations=0`, which never reaches cv, so we drive a real cv failure
# instead). Candidate `bagging_temperature=0.0` trains fine.
_DEGENERATE_GRID = {"bagging_temperature": [0.0, 1e9]}


def _bayesian_reg():
    return catboost_rs.CatBoostRegressor(
        loss_function="RMSE",
        iterations=10,
        learning_rate=0.2,
        random_seed=0,
        bootstrap_type="Bayesian",
    )


def test_grid_search_error_score_nan_warns_and_continues():
    # With the sklearn-default `error_score=np.nan`, the degenerate candidate is
    # ranked WORST, the search CONTINUES, and a FitFailedWarning is raised.
    x, y = _toy_regression()
    folds = _fixed_folds()
    with pytest.warns(catboost_rs.FitFailedWarning):
        res = catboost_rs.grid_search(
            _bayesian_reg(),
            _DEGENERATE_GRID,
            x,
            y,
            cv=3,
            folds=folds,
            shuffle=False,
            refit=False,
        )
    assert set(res.keys()) == {"params", "cv_results"}
    # The winner is the NON-degenerate point, never bagging_temperature=1e9.
    assert res["params"] == {"bagging_temperature": 0.0}


def test_grid_search_error_score_raise():
    # Same degenerate grid, but `error_score='raise'` re-raises the failure.
    x, y = _toy_regression()
    folds = _fixed_folds()
    with pytest.raises(catboost_rs.CatBoostError):
        catboost_rs.grid_search(
            _bayesian_reg(),
            _DEGENERATE_GRID,
            x,
            y,
            cv=3,
            folds=folds,
            shuffle=False,
            refit=False,
            error_score="raise",
        )


def test_grid_search_error_score_numeric():
    # A FINITE numeric error_score competes normally (participates in ranking),
    # unlike NaN (skipped). Two facets:
    #
    # (a) error_score=1e9 (worse than any real RMSE on the min-optimal metric):
    #     the degenerate candidate competes but LOSES, so the search completes
    #     with a valid winner and warns.
    x, y = _toy_regression()
    folds = _fixed_folds()
    with pytest.warns(catboost_rs.FitFailedWarning):
        res = catboost_rs.grid_search(
            _bayesian_reg(),
            _DEGENERATE_GRID,
            x,
            y,
            cv=3,
            folds=folds,
            shuffle=False,
            refit=False,
            error_score=1e9,
        )
    assert res["params"] == {"bagging_temperature": 0.0}

    # (b) error_score=0.0 (LOWER than any real positive RMSE): the finite score
    #     out-competes every real candidate and the degenerate candidate is
    #     SELECTED — proving a finite error_score competes. Because THIS failure
    #     is a cv-error (the candidate produced no cv result at all), selecting it
    #     surfaces the typed "selected candidate produced no cv result" guard
    #     (mapped to CatBoostError) rather than a NaN-bearing result. (Contrast
    #     the Rust `error_score_numeric_competes` test, whose iterations(0)
    #     candidate yields an Ok-but-empty column, so a numeric winner returns
    #     Ok.)
    with pytest.raises(catboost_rs.CatBoostError, match="no cv result"):
        catboost_rs.grid_search(
            _bayesian_reg(),
            _DEGENERATE_GRID,
            x,
            y,
            cv=3,
            folds=folds,
            shuffle=False,
            refit=False,
            error_score=0.0,
        )


def test_grid_search_error_score_bad_string_raises():
    # A non-"raise" string is a value error (sklearn accepts only 'raise' or a
    # number).
    x, y = _toy_regression()
    folds = _fixed_folds()
    reg = catboost_rs.CatBoostRegressor(loss_function="RMSE", iterations=10)
    with pytest.raises(catboost_rs.CatBoostValueError):
        catboost_rs.grid_search(
            reg,
            {"depth": [3, 5]},
            x,
            y,
            cv=3,
            folds=folds,
            shuffle=False,
            refit=False,
            error_score="nonsense",
        )


def test_grid_search_refit_fits_estimator():
    x, y = _toy_regression()
    folds = _fixed_folds()
    reg = catboost_rs.CatBoostRegressor(
        loss_function="RMSE", iterations=15, learning_rate=0.2, random_seed=0
    )
    grid = {"depth": [3, 5]}
    res = catboost_rs.grid_search(
        reg, grid, x, y, cv=3, folds=folds, shuffle=False, refit=True
    )
    # After refit the estimator predicts (it was fit in place with best params).
    preds = reg.predict(x)
    assert preds.shape == (x.shape[0],)
    # And its params now reflect the winning grid point.
    assert reg.get_params()["depth"] == res["params"]["depth"]
