"""ORCH-02-S6 — `catboost_rs.randomized_search` determinism / subset / shape.

Mirrors the supported subset of upstream `CatBoost.randomized_search(
param_distributions, X, y, cv=3, n_iter=10, ...)`: expand the (discrete)
`dict[str, list]` distributions into the full Cartesian product, then let the
Rust facade deterministically subsample `min(n_iter, product)` candidates via
the project's own `TFastRng64` (seeded by `partition_random_seed`), evaluate
each by cv, and return the same `{"params","cv_results"}` dict as
`grid_search`.

All randomness lives in the seeded Rust RNG, so the SAME seed => the SAME
result (determinism), and `n_iter >= product_size` reduces to `grid_search`.
"""

import numpy as np
import pytest

import catboost_rs


def _toy_regression(n=48, k=4, seed=11):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, k)).astype(np.float32)
    y = (x[:, 0] * 2.0 - x[:, 1] + 0.5 * x[:, 2] ** 2).astype(np.float32)
    return x, y


def _fixed_folds(n=48, k=3):
    folds = []
    for f in range(k):
        lo = f * n // k
        hi = (f + 1) * n // k
        folds.append(list(range(lo, hi)))
    return folds


def _twelve_point_grid():
    # 4 x 3 = 12 discrete grid points.
    return {"depth": [2, 4, 6, 8], "l2_leaf_reg": [1.0, 3.0, 5.0]}


def test_randomized_search_symbol_resolves():
    assert hasattr(catboost_rs, "randomized_search")
    assert callable(catboost_rs.randomized_search)


def test_randomized_search_shape_and_subset():
    x, y = _toy_regression()
    folds = _fixed_folds()
    reg = catboost_rs.CatBoostRegressor(
        loss_function="RMSE", iterations=12, learning_rate=0.2, random_seed=0
    )
    res = catboost_rs.randomized_search(
        reg,
        _twelve_point_grid(),
        x,
        y,
        cv=3,
        n_iter=2,
        partition_random_seed=0,
        folds=folds,
        shuffle=False,
        refit=False,
    )
    assert set(res.keys()) == {"params", "cv_results"}
    assert "test-RMSE-mean" in res["cv_results"]
    # The winning params must be a valid grid point.
    valid = [
        {"depth": d, "l2_leaf_reg": r}
        for d in [2, 4, 6, 8]
        for r in [1.0, 3.0, 5.0]
    ]
    assert res["params"] in valid


def test_randomized_search_determinism_same_seed():
    x, y = _toy_regression()
    folds = _fixed_folds()

    def run():
        reg = catboost_rs.CatBoostRegressor(
            loss_function="RMSE", iterations=12, learning_rate=0.2, random_seed=0
        )
        return catboost_rs.randomized_search(
            reg,
            _twelve_point_grid(),
            x,
            y,
            cv=3,
            n_iter=5,
            partition_random_seed=0,
            folds=folds,
            shuffle=False,
            refit=False,
        )

    a = run()
    b = run()
    assert a["params"] == b["params"]
    ga = np.asarray(a["cv_results"]["test-RMSE-mean"], dtype=np.float64)
    gb = np.asarray(b["cv_results"]["test-RMSE-mean"], dtype=np.float64)
    assert np.array_equal(ga, gb)


# A degenerate grid point: extreme Bayesian `bagging_temperature` so cv errors
# for that candidate (the Python validator rejects the simpler iterations=0).
_DEGENERATE_GRID = {"bagging_temperature": [0.0, 1e9]}


def _bayesian_reg():
    return catboost_rs.CatBoostRegressor(
        loss_function="RMSE",
        iterations=10,
        learning_rate=0.2,
        random_seed=0,
        bootstrap_type="Bayesian",
    )


def test_randomized_search_error_score_raise_determinism():
    # `error_score='raise'` over a grid that includes a degenerate point
    # re-raises when that point is among the sampled subset. With n_iter large
    # enough to sample the whole product, the degenerate point is always
    # evaluated, so 'raise' must abort deterministically.
    x, y = _toy_regression()
    folds = _fixed_folds()
    with pytest.raises(catboost_rs.CatBoostError):
        catboost_rs.randomized_search(
            _bayesian_reg(),
            _DEGENERATE_GRID,
            x,
            y,
            cv=3,
            n_iter=100,  # >= product => samples every point incl. the degenerate
            partition_random_seed=0,
            folds=folds,
            shuffle=False,
            refit=False,
            error_score="raise",
        )


def test_randomized_search_error_score_nan_warns():
    # Default `error_score=np.nan`: the degenerate point is ranked worst; a
    # FitFailedWarning is raised and the valid non-degenerate point wins.
    x, y = _toy_regression()
    folds = _fixed_folds()
    with pytest.warns(catboost_rs.FitFailedWarning):
        res = catboost_rs.randomized_search(
            _bayesian_reg(),
            _DEGENERATE_GRID,
            x,
            y,
            cv=3,
            n_iter=100,
            partition_random_seed=0,
            folds=folds,
            shuffle=False,
            refit=False,
        )
    assert res["params"] == {"bagging_temperature": 0.0}


def test_randomized_search_full_when_niter_ge_grid():
    x, y = _toy_regression()
    folds = _fixed_folds()
    grid = _twelve_point_grid()
    base_kwargs = dict(
        loss_function="RMSE", iterations=12, learning_rate=0.2, random_seed=0
    )

    reg_r = catboost_rs.CatBoostRegressor(**base_kwargs)
    rand = catboost_rs.randomized_search(
        reg_r,
        grid,
        x,
        y,
        cv=3,
        n_iter=100,  # >= 12 -> evaluates the whole product
        partition_random_seed=0,
        folds=folds,
        shuffle=False,
        refit=False,
    )

    reg_g = catboost_rs.CatBoostRegressor(**base_kwargs)
    full = catboost_rs.grid_search(
        reg_g, grid, x, y, cv=3, folds=folds, shuffle=False, refit=False
    )

    # n_iter >= grid_size behaves like grid_search: same best params.
    assert rand["params"] == full["params"]
    gr = np.asarray(rand["cv_results"]["test-RMSE-mean"], dtype=np.float64)
    gf = np.asarray(full["cv_results"]["test-RMSE-mean"], dtype=np.float64)
    assert np.max(np.abs(gr - gf)) <= 1e-5
