"""F16 / F17 / F20 — the public Python categorical surface.

The six categorical / CTR params moved from REJECTED-as-a-parity-gap to
IMPLEMENTED, `cat_features` became a `fit()` kwarg, and the whole path is
gated end to end against the frozen upstream one-hot oracle.
"""

from pathlib import Path

import numpy as np
import pytest

import catboost_rs
from catboost_rs import CatBoostClassifier, CatBoostParameterError, CatBoostRegressor

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FIXTURES = _REPO_ROOT / "crates" / "cb-oracle" / "fixtures"


def _toy_xy():
    x = np.array(
        [[0.0, 1.0], [1.0, 0.0], [2.0, 2.0], [3.0, 1.0]], dtype=np.float32
    )
    y = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    return x, y


# ---------------------------------------------------------------------------
# F15 / F16 — the promoted params are IMPLEMENTED, not parity gaps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "cat_features",
        "one_hot_max_size",
        "max_ctr_complexity",
        "simple_ctr",
        "combinations_ctr",
        "counter_calc_method",
    ],
)
def test_promoted_params_report_implemented(name):
    assert catboost_rs._param_status(name) == "IMPLEMENTED"


def test_nan_mode_is_still_a_rejected_parity_gap():
    """The promotion must not have loosened the honesty policy wholesale."""
    assert catboost_rs._param_status("nan_mode") == "KNOWN_NOT_YET"
    x, y = _toy_xy()
    with pytest.raises(CatBoostParameterError):
        CatBoostRegressor(nan_mode="Min").fit(x, y)


def test_scalar_categorical_params_are_accepted_at_fit():
    x, y = _toy_xy()
    model = CatBoostRegressor(
        iterations=3,
        depth=2,
        one_hot_max_size=4,
        max_ctr_complexity=2,
        counter_calc_method="Full",
    )
    model.fit(x, y)  # must not raise
    assert model.predict(x).shape == (4,)


def test_cpu_illegal_ctr_types_are_rejected_by_name():
    """F16's inverted assertion.

    T13 originally expected ``"FeatureFreq"`` in the ACCEPTED variants list.
    That was wrong: `FloatTargetMeanValue` and `FeatureFreq` are GPU-only
    (`catboost_options.cpp:504-509`) and upstream rejects them on CPU, exactly
    as this crate's engine-side E02 guard does.
    """
    for bad in ("FloatTargetMeanValue", "FeatureFreq"):
        with pytest.raises(CatBoostParameterError) as e:
            CatBoostRegressor(simple_ctr=[bad]).fit(*_toy_xy())
        msg = str(e.value)
        assert bad in msg
        assert "not implemented on CPU" in msg
        for ok in ("Borders", "Buckets", "BinarizedTargetMeanValue", "Counter"):
            assert ok in msg


def test_multiple_ctr_descriptions_are_rejected_naming_the_parity_gap():
    with pytest.raises(CatBoostParameterError) as e:
        CatBoostRegressor(
            simple_ctr=["Borders:Prior=0.5", "Counter:Prior=0"]
        ).fit(*_toy_xy())
    assert "one CTR description" in str(e.value)


def test_non_unit_prior_denominator_is_rejected():
    with pytest.raises(CatBoostParameterError) as e:
        CatBoostRegressor(simple_ctr=["Borders:Prior=1/2"]).fit(*_toy_xy())
    assert "denominator" in str(e.value)


def test_empty_combinations_ctr_is_accepted_and_documented():
    """The recorded ``combinations_ctr=[]`` mapping (option (a)).

    The engine has no "disabled" representation for the scalar CTR type, so
    ``[]`` maps to ``max_ctr_complexity = 1`` — the only in-engine way to
    suppress combination CTRs. That the mapping ACTUALLY moves the value (not
    merely that ``fit()`` stays quiet) is asserted on the Rust side by
    ``empty_combinations_ctr_maps_to_max_ctr_complexity_one``; here we only
    check the kwarg is accepted, since the builder is not visible from Python.
    """
    x, y = _toy_xy()
    CatBoostRegressor(iterations=3, depth=2, combinations_ctr=[]).fit(x, y)


def test_empty_simple_ctr_is_rejected_not_ignored():
    with pytest.raises(CatBoostParameterError) as e:
        CatBoostRegressor(simple_ctr=[]).fit(*_toy_xy())
    assert "cannot be disabled" in str(e.value)


# ---------------------------------------------------------------------------
# F17 — `cat_features` as a fit() kwarg
# ---------------------------------------------------------------------------


def _pandas_frame_with_cat():
    pd = pytest.importorskip("pandas")
    return pd.DataFrame(
        {
            "f0": np.arange(40, dtype=np.float32) % 7,
            "f1": np.arange(40, dtype=np.float32) % 3,
            "c0": [f"c{i % 9}" for i in range(40)],
        }
    )


def test_cat_features_fit_kwarg_trains_and_predicts():
    df = _pandas_frame_with_cat()
    y = np.array([float(i % 2) for i in range(40)], dtype=np.float32)

    model = CatBoostClassifier(
        iterations=3,
        depth=2,
        learning_rate=0.1,
        one_hot_max_size=1,
        max_ctr_complexity=1,
        boost_from_average=False,
        random_strength=0,
    )
    model.fit(df, y, cat_features=[2])
    preds = model.predict(df)
    assert preds.shape == (40,)


def test_cat_features_on_a_numpy_matrix_is_a_typed_error_not_a_silent_drop():
    """Finding F2: ``ingest_to_owned``'s NumPy branch DROPS ``cat_features``.

    Without the post-ingestion width guard the user would train a float-only
    model and never learn the argument did nothing.
    """
    x = np.arange(40 * 3, dtype=np.float32).reshape(40, 3)
    y = np.array([float(i % 2) for i in range(40)], dtype=np.float32)
    with pytest.raises(Exception) as e:
        CatBoostRegressor(iterations=3, depth=2).fit(x, y, cat_features=[2])
    assert "cat_features declared" in str(e.value)


def test_duplicate_cat_feature_index_is_reported_as_a_duplicate():
    """MINOR-11: ``cat_features=[2, 2]`` must name the DUPLICATE, not
    mis-report it as a width mismatch ("declared 2 ... carries 1")."""
    df = _pandas_frame_with_cat()
    y = np.array([float(i % 2) for i in range(40)], dtype=np.float32)
    with pytest.raises(Exception) as e:
        CatBoostRegressor(iterations=3, depth=2).fit(df, y, cat_features=[2, 2])
    assert "duplicate" in str(e.value).lower()


def test_cat_features_with_a_pool_raises_upstream_exact():
    """OQ-3, upstream-exact (`core.py:1522-1533`): a Pool already declares its
    categorical columns, so combining it with ``cat_features`` is ambiguous."""
    df = _pandas_frame_with_cat()
    y = np.array([float(i % 2) for i in range(40)], dtype=np.float32)
    pool = catboost_rs.Pool(df, y, cat_features=[2])
    with pytest.raises(Exception) as e:
        CatBoostRegressor(iterations=3, depth=2).fit(pool, cat_features=[2])
    assert "cat_features cannot be given when the data is a Pool" in str(e.value)


def test_cat_features_works_on_all_three_estimators():
    df = _pandas_frame_with_cat()
    y = np.array([float(i % 2) for i in range(40)], dtype=np.float32)
    for cls in (CatBoostRegressor, CatBoostClassifier):
        model = cls(iterations=3, depth=2, one_hot_max_size=1, max_ctr_complexity=1)
        model.fit(df, y, cat_features=[2])
        assert model.predict(df).shape == (40,)


# ---------------------------------------------------------------------------
# F20 (SPEC-CATF-16, acceptance A11) — the PUBLIC PYTHON API oracle, <= 1e-5
# ---------------------------------------------------------------------------


def _one_hot_fixture(scenario="default_binary"):
    import json

    d = _FIXTURES / "one_hot_train" / scenario
    cfg = json.loads((d / "config.json").read_text())
    cat = json.loads((d / "cat_cols.json").read_text())
    return (
        cfg,
        np.load(d / "X_float.npy"),
        np.load(d / "y.npy"),
        cat["columns"],
        cat["cat_feature_indices_in_x_all"],
        np.load(d / "predictions.npy"),
    )


def test_public_api_one_hot_oracle_parity():
    """Drive the frozen upstream one-hot corpus end to end through the PUBLIC
    Python API and match ``predictions.npy`` at ``atol=1e-5, rtol=0``.

    Localization ladder, if this fails (STOP AND REPORT at the first hit):
      1. ``crates/cb-train/tests/one_hot_oracle_test.rs`` IS the direct Rust
         ``train_cat`` + ``predict_raw_cat`` comparison on this same corpus. If
         it passes and this fails, the gap is in the facade/binding wiring.
      2. If the Rust-direct path also diverges, the gap is inside the shipped
         engine — STOP AND REPORT.
      3. If it passes at ``max_ctr_complexity=0`` and fails at ``> 1``, that is
         the ORD-06/ORD-07 combination-CTR gating bug
         (`.planning/phases/24-ctr-split-search-correctness/`) — STOP AND
         REPORT, do NOT fix it here. The ``permutation_count=4`` risk is
         already closed by ``multi_permutation_e2e_oracle_test.rs``.
    """
    pd = pytest.importorskip("pandas")
    cfg, x_float, y, cat_columns, cat_idx, expected = _one_hot_fixture()
    p = cfg["params"]

    frame = {f"f{i}": x_float[:, i].astype(np.float32) for i in range(x_float.shape[1])}
    for j, col in enumerate(cat_columns):
        frame[f"c{j}"] = col
    df = pd.DataFrame(frame)

    # CatBoostRegressor, not CatBoostClassifier: the fixture's reference vector
    # is RawFormulaVal, and the regressor's `predict` returns exactly that
    # (the classifier's returns CLASS labels). `loss_function="Logloss"` keeps
    # the TRAINING objective identical to the fixture's.
    model = CatBoostRegressor(
        iterations=p["iterations"],
        depth=p["depth"],
        learning_rate=p["learning_rate"],
        l2_leaf_reg=p["l2_leaf_reg"],
        loss_function=p["loss_function"],
        leaf_estimation_method=p["leaf_estimation_method"],
        bootstrap_type=p["bootstrap_type"],
        boost_from_average=p["boost_from_average"],
        random_strength=p["random_strength"],
        random_seed=p["random_seed"],
        score_function=p["score_function"],
        one_hot_max_size=p["one_hot_max_size"],
        max_ctr_complexity=p["max_ctr_complexity"],
    )
    model.fit(df, y.astype(np.float32), cat_features=cat_idx)
    actual = model.predict(df)

    # Non-degeneracy: a constant prediction vector would match a broken model
    # that had simply learned the base rate.
    assert actual.std() > 1e-6, f"predictions are degenerate (std {actual.std()})"

    # The configuration must genuinely have produced a categorical split, so a
    # run that degenerated to float-only cannot pass vacuously. Proof through
    # the PUBLIC API: after F10/F12, predicting WITHOUT the categorical column
    # raises iff `needs_cat_columns()` is true. With `max_ctr_complexity=0`
    # there is no CTR data, so the only thing that can make it true is a
    # `ModelSplit::OneHot`.
    float_only = df[[c for c in df.columns if c.startswith("f")]]
    with pytest.raises(Exception) as e:
        model.predict(float_only)
    assert "categorical" in str(e.value), (
        "the fitted model carries no categorical split — this configuration "
        f"degenerated to float-only and the oracle comparison is vacuous: {e.value}"
    )

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)
