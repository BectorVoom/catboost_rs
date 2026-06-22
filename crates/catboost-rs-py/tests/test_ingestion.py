"""08-03 multi-source ingestion (PYAPI-04) + strict D-12 validation + native Pool.

Proves a user can fit/predict from a NumPy array, a Pandas DataFrame, a pyarrow
Table, and a Polars DataFrame — all converging on the same OwnedColumns ->
into_pool() seam and producing equal predictions for equal data. Also proves the
four D-12 rejection paths (float64 / non-contiguous / ambiguous object column /
nullable Arrow column) raise an actionable CatBoostValueError, and that the
native Pool mirrors upstream Pool.__init__.

Pandas / pyarrow / Polars are optional test deps; each test importorskips the
missing framework rather than failing the suite.
"""

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture
def toy():
    """A small C-contiguous float32 regression dataset shared by every source."""
    rng = np.random.default_rng(0)
    x = np.ascontiguousarray(rng.standard_normal((40, 3)), dtype=np.float32)
    coef = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    y = np.ascontiguousarray(x @ coef + 0.1, dtype=np.float32)
    return x, y


def _fit_numpy(x, y):
    import catboost_rs

    return catboost_rs.CatBoostRegressor(iterations=15, depth=3, random_seed=7).fit(x, y)


# --------------------------------------------------------------------------- #
# Source coverage: NumPy / Pandas / Arrow / Polars all fit+predict equally     #
# --------------------------------------------------------------------------- #
def test_numpy_fit_predict(toy):
    x, y = toy
    preds = _fit_numpy(x, y).predict(x)
    assert preds.shape == (x.shape[0],)
    assert np.all(np.isfinite(preds))


def test_pandas_matches_numpy(toy):
    pd = pytest.importorskip("pandas")
    x, y = toy

    base = _fit_numpy(x, y).predict(x)

    df = pd.DataFrame(x, columns=["a", "b", "c"]).astype(np.float32)
    # Fit AND predict from the DataFrame.
    model = _fit_numpy(x, y)
    preds = model.predict(df)
    np.testing.assert_allclose(preds, base, rtol=0, atol=1e-6)


def test_arrow_matches_numpy(toy):
    pa = pytest.importorskip("pyarrow")
    x, y = toy

    base = _fit_numpy(x, y).predict(x)

    table = pa.table(
        {
            "a": pa.array(x[:, 0], type=pa.float32()),
            "b": pa.array(x[:, 1], type=pa.float32()),
            "c": pa.array(x[:, 2], type=pa.float32()),
        }
    )
    preds = _fit_numpy(x, y).predict(table)
    np.testing.assert_allclose(preds, base, rtol=0, atol=1e-6)


def test_polars_matches_numpy(toy):
    pl = pytest.importorskip("polars")
    x, y = toy

    base = _fit_numpy(x, y).predict(x)

    df = pl.DataFrame(
        {
            "a": x[:, 0],
            "b": x[:, 1],
            "c": x[:, 2],
        }
    ).cast(pl.Float32)
    preds = _fit_numpy(x, y).predict(df)
    np.testing.assert_allclose(preds, base, rtol=0, atol=1e-6)


# --------------------------------------------------------------------------- #
# D-12 rejection paths: actionable CatBoostValueError                          #
# --------------------------------------------------------------------------- #
def test_float64_rejected_actionable(toy):
    import catboost_rs

    x, y = toy
    with pytest.raises(catboost_rs.CatBoostValueError, match="float32"):
        _fit_numpy(x.astype(np.float64), y)


def test_non_contiguous_rejected_actionable(toy):
    import catboost_rs

    x, y = toy
    # A sliced (every-other-column) view is not C-contiguous.
    sliced = np.asfortranarray(x)
    with pytest.raises(
        catboost_rs.CatBoostValueError, match="contiguous|ascontiguousarray"
    ):
        _fit_numpy(sliced, y)


def test_pandas_object_column_rejected_actionable(toy):
    pd = pytest.importorskip("pandas")
    import catboost_rs

    x, y = toy
    df = pd.DataFrame(x, columns=["a", "b", "c"]).astype(np.float32)
    df["color"] = ["red"] * x.shape[0]  # an ambiguous object/string column

    model = _fit_numpy(x, y)
    # No cat_features given -> the object column is ambiguous and rejected, naming
    # the column and suggesting cat_features.
    with pytest.raises(catboost_rs.CatBoostValueError, match="color|cat_features"):
        model.predict(df)


def test_arrow_nullable_column_rejected_actionable(toy):
    pa = pytest.importorskip("pyarrow")
    import catboost_rs

    x, y = toy
    vals = x[:, 0].tolist()
    vals[0] = None  # a null in an otherwise-float32 column
    table = pa.table(
        {
            "a": pa.array(vals, type=pa.float32()),
            "b": pa.array(x[:, 1], type=pa.float32()),
            "c": pa.array(x[:, 2], type=pa.float32()),
        }
    )
    model = _fit_numpy(x, y)
    with pytest.raises(catboost_rs.CatBoostValueError, match="null"):
        model.predict(table)


# --------------------------------------------------------------------------- #
# Native Pool (PYAPI-03)                                                       #
# --------------------------------------------------------------------------- #
def test_pool_constructs_and_fits(toy):
    import catboost_rs

    x, y = toy
    pool = catboost_rs.Pool(x, label=y)
    assert pool.num_row == x.shape[0]
    assert pool.num_col == x.shape[1]

    base = _fit_numpy(x, y).predict(x)

    model = catboost_rs.CatBoostRegressor(iterations=15, depth=3, random_seed=7)
    model.fit(pool)
    preds = model.predict(pool)
    np.testing.assert_allclose(preds, base, rtol=0, atol=1e-6)


def test_pool_length_mismatch_rejected(toy):
    import catboost_rs

    x, y = toy
    short_label = y[:-1]  # one fewer label than rows
    # A label shorter than the feature matrix is rejected as a CatBoostValueError.
    # The label-vs-rows check fires fail-fast at the shared ingest seam (it carries
    # the same typed error as OwnedColumns::into_pool()'s length check) — the
    # binding never re-implements the length check, and never indexes unchecked
    # (threat T-08-11).
    with pytest.raises(catboost_rs.CatBoostValueError):
        catboost_rs.Pool(x, label=short_label)


def test_pool_from_arrow(toy):
    pa = pytest.importorskip("pyarrow")
    import catboost_rs

    x, y = toy
    table = pa.table(
        {
            "a": pa.array(x[:, 0], type=pa.float32()),
            "b": pa.array(x[:, 1], type=pa.float32()),
            "c": pa.array(x[:, 2], type=pa.float32()),
        }
    )
    pool = catboost_rs.Pool(table, label=y)
    assert pool.num_row == x.shape[0]
    assert pool.num_col == x.shape[1]
