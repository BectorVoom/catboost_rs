"""PARAM-03 — the class-weight family and `ignored_features` from Python.

`class_weights` / `auto_class_weights` / `scale_pos_weight` needed no new engine
code: `cb_data::weights` already carried the upstream-faithful computation, gated
against the frozen `class_weights/` fixture by `cb-data`'s `weights_oracle_test`.
What was missing was the path that APPLIES the resolved per-object weights during
a fit. `ignored_features` is genuinely new.

These tests own the Python surface: the kwargs are accepted, they change the fit,
the mutually-exclusive ones are refused together, and each rejection names the
offending parameter. The weight VALUES are the Rust oracle test's job.
"""

import numpy as np
import pytest

from catboost_rs import (
    CatBoostClassifier,
    CatBoostParameterError,
    CatBoostRegressor,
)

N = 160


def _imbalanced_binary():
    """Feature 0 predicts the label; class 1 is a 1-in-4 minority."""
    rng = np.random.default_rng(3)
    x = np.ascontiguousarray(
        np.column_stack(
            [
                np.arange(N, dtype=np.float32),
                rng.random(N).astype(np.float32),
            ]
        ),
        dtype=np.float32,
    )
    y = (np.arange(N) % 4 == 0).astype(np.float32)
    return x, y


def _regression():
    x = np.ascontiguousarray(
        np.arange(N, dtype=np.float32).reshape(-1, 1), dtype=np.float32
    )
    y = (x[:, 0] * 0.5 + 1.0).astype(np.float32)
    return x, y


def _clf(**kwargs):
    return CatBoostClassifier(iterations=20, depth=3, learning_rate=0.2, **kwargs)


# ─── the class-weight family ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs",
    [
        {"class_weights": [1.0, 5.0]},
        {"auto_class_weights": "Balanced"},
        {"auto_class_weights": "SqrtBalanced"},
        {"scale_pos_weight": 4.0},
    ],
)
def test_class_weight_controls_change_the_model(kwargs):
    x, y = _imbalanced_binary()
    base = _clf()
    base.fit(x, y)
    tuned = _clf(**kwargs)
    tuned.fit(x, y)
    assert not np.allclose(base.predict_proba(x), tuned.predict_proba(x)), (
        f"{kwargs} was accepted but never reached the trainer"
    )


def test_scale_pos_weight_equals_the_explicit_two_element_vector():
    """`scale_pos_weight=w` is defined as `class_weights=[1, w]` — asserted as an
    equality rather than as two separately-plausible behaviours."""
    x, y = _imbalanced_binary()
    scaled = _clf(scale_pos_weight=4.0)
    scaled.fit(x, y)
    explicit = _clf(class_weights=[1.0, 4.0])
    explicit.fit(x, y)
    np.testing.assert_array_equal(scaled.predict_proba(x), explicit.predict_proba(x))


def test_unit_class_weights_are_the_identity():
    """Active as a parameter, but multiplies every weight by one.

    Separates "applied" from "perturbs something": an implementation that
    REPLACED the pool weights instead of multiplying would pass the change test
    above and fail here.
    """
    x, y = _imbalanced_binary()
    base = _clf()
    base.fit(x, y)
    unit = _clf(class_weights=[1.0, 1.0])
    unit.fit(x, y)
    np.testing.assert_array_equal(base.predict_proba(x), unit.predict_proba(x))


def test_the_two_auto_schemes_differ_from_each_other():
    x, y = _imbalanced_binary()
    balanced = _clf(auto_class_weights="Balanced")
    balanced.fit(x, y)
    sqrt = _clf(auto_class_weights="SqrtBalanced")
    sqrt.fit(x, y)
    assert not np.allclose(balanced.predict_proba(x), sqrt.predict_proba(x)), (
        "Balanced (max/w) and SqrtBalanced (sqrt(max/w)) must not resolve alike"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"class_weights": [1.0, 2.0], "auto_class_weights": "Balanced"},
        {"class_weights": [1.0, 2.0], "scale_pos_weight": 3.0},
        {"auto_class_weights": "Balanced", "scale_pos_weight": 3.0},
    ],
)
def test_the_class_weight_controls_are_mutually_exclusive(kwargs):
    """All three write the same per-object weight; any precedence rule would
    silently discard one the caller set on purpose."""
    x, y = _imbalanced_binary()
    with pytest.raises(CatBoostParameterError) as excinfo:
        _clf(**kwargs).fit(x, y)
    assert "at most one" in str(excinfo.value)


def test_a_class_weight_control_on_a_regression_loss_is_rejected():
    x, y = _regression()
    with pytest.raises(CatBoostParameterError) as excinfo:
        CatBoostRegressor(iterations=5, class_weights=[1.0, 2.0]).fit(x, y)
    assert "classification" in str(excinfo.value)


def test_an_unknown_auto_class_weights_value_is_rejected_by_name():
    x, y = _imbalanced_binary()
    with pytest.raises(CatBoostParameterError) as excinfo:
        _clf(auto_class_weights="Bananas").fit(x, y)
    assert "Bananas" in str(excinfo.value)


def test_too_few_class_weights_is_rejected():
    x, y = _imbalanced_binary()
    with pytest.raises(CatBoostParameterError) as excinfo:
        _clf(class_weights=[1.0]).fit(x, y)
    assert "2 classes" in str(excinfo.value)


@pytest.mark.parametrize("bad", [{"scale_pos_weight": 0.0}, {"class_weights": [1.0, -2.0]}])
def test_out_of_range_class_weights_are_rejected(bad):
    x, y = _imbalanced_binary()
    with pytest.raises(CatBoostParameterError):
        _clf(**bad).fit(x, y)


# ─── ignored_features ────────────────────────────────────────────────────────


def test_ignoring_the_predictive_feature_degrades_the_fit():
    """Feature 0 carries the signal; ignoring it must measurably hurt.

    Asserted as a comparison against the unrestricted control, so a no-op
    implementation (parameter accepted, feature still split on) fails.
    """
    x, y = _imbalanced_binary()
    full = _clf()
    full.fit(x, y)
    restricted = _clf(ignored_features=[0])
    restricted.fit(x, y)

    full_err = float(np.mean((full.predict(x) - y) ** 2))
    restricted_err = float(np.mean((restricted.predict(x) - y) ** 2))
    assert restricted_err > full_err, (
        f"ignoring the predictive feature must hurt; got restricted={restricted_err} "
        f"full={full_err}"
    )


def test_ignored_features_preserves_the_predict_width():
    """The ignored feature keeps its index, so predict still takes the full
    matrix — the property that motivated emptying its borders rather than
    dropping the column."""
    x, y = _imbalanced_binary()
    model = _clf(ignored_features=[0])
    model.fit(x, y)
    assert model.predict(x).shape == (N,)


def test_an_empty_ignored_features_list_is_a_no_op():
    x, y = _imbalanced_binary()
    base = _clf()
    base.fit(x, y)
    empty = _clf(ignored_features=[])
    empty.fit(x, y)
    np.testing.assert_array_equal(base.predict_proba(x), empty.predict_proba(x))


def test_an_out_of_range_ignored_feature_index_is_rejected():
    """A typo that silently ignores nothing defeats the parameter's purpose."""
    x, y = _imbalanced_binary()
    with pytest.raises(CatBoostParameterError) as excinfo:
        _clf(ignored_features=[7]).fit(x, y)
    msg = str(excinfo.value)
    assert "out of range" in msg and "7" in msg, msg
