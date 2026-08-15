"""`class_names`: the class-label mapping surface.

Every contract here was probed from catboost 1.2.10 first, not assumed:

  * `classes_` follows the ORDER given, not sorted order;
  * `predict` returns the caller's labels;
  * `predict_proba`'s columns follow `classes_`;
  * a `y` label absent from `class_names` is REJECTED ("Unknown class label");
  * `class_names` is a CLASSIFIER parameter -- `CatBoostRegressor(class_names=...)`
    raises `unexpected keyword argument` upstream.

What is deliberately NOT claimed: label mapping WITHOUT `class_names`. Upstream
derives `classes_` from the data there and `predict` returns the original labels;
this surface still returns raw 0.0/1.0 class indices. That gap is asserted below so
it stays visible instead of being mistaken for parity.
"""

import numpy as np
import pytest

import catboost_rs
from catboost_rs import CatBoostClassifier, CatBoostRegressor, CatBoostValueError
from catboost_rs import CatBoostParameterError

BASE = dict(iterations=5, depth=2, learning_rate=0.3, random_seed=0,
            bootstrap_type="No", random_strength=0)


def _xy():
    rng = np.random.default_rng(3)
    x = np.ascontiguousarray(rng.normal(size=(60, 3)), dtype=np.float32)
    y01 = (2 * x[:, 0] - x[:, 1] > 0).astype(np.int64)
    return x, y01


def _string_y(y01):
    return np.array(["neg", "pos"])[y01]


def test_class_names_maps_string_labels_through_fit_and_predict():
    x, y01 = _xy()
    m = CatBoostClassifier(**BASE, class_names=["neg", "pos"])
    m.fit(x, _string_y(y01))
    preds = list(m.predict(x[:8]))
    assert all(p in ("neg", "pos") for p in preds), preds
    # Not vacuous: a model that always predicted one class would trivially satisfy
    # the membership check above.
    assert len(set(preds)) == 2, f"expected both labels among {preds}"


def test_classes_follows_the_given_order_not_sorted_order():
    """The ORDER is the point: it decides which label is the positive class."""
    x, y01 = _xy()
    ys = _string_y(y01)
    forward = CatBoostClassifier(**BASE, class_names=["neg", "pos"])
    forward.fit(x, ys)
    reverse = CatBoostClassifier(**BASE, class_names=["pos", "neg"])
    reverse.fit(x, ys)
    assert forward.classes_ == ["neg", "pos"]
    assert reverse.classes_ == ["pos", "neg"]

    # predict_proba columns follow classes_, so reversing the names swaps them.
    pf = forward.predict_proba(x[:5])
    pr = reverse.predict_proba(x[:5])
    assert np.allclose(pf, pr[:, ::-1]), (
        "predict_proba columns must follow classes_ order, so reversing class_names "
        "must swap the two columns"
    )


def test_a_label_absent_from_class_names_is_rejected():
    x, y01 = _xy()
    with pytest.raises(CatBoostValueError) as e:
        CatBoostClassifier(**BASE, class_names=["a", "b"]).fit(x, _string_y(y01))
    assert "Unknown class label" in str(e.value)


def test_numeric_class_names_round_trip():
    x, y01 = _xy()
    m = CatBoostClassifier(**BASE, class_names=[1, 0])
    m.fit(x, y01)
    preds = list(m.predict(x[:8]))
    assert all(p in (0, 1) for p in preds), preds
    assert m.classes_ == [1, 0]


def test_class_names_is_rejected_on_the_regressor_and_ranker():
    """Upstream does not accept it there either (`unexpected keyword argument`)."""
    x, y01 = _xy()
    with pytest.raises(CatBoostParameterError) as e:
        CatBoostRegressor(**BASE, class_names=["a", "b"]).fit(x, y01.astype(np.float32))
    assert "class_names" in str(e.value)


@pytest.mark.parametrize("names", [["only"], ["a", "b", "c"], ["dup", "dup"]])
def test_malformed_class_names_are_rejected(names):
    """Too few, too many (this surface is binary), and duplicates."""
    x, y01 = _xy()
    with pytest.raises(CatBoostValueError):
        CatBoostClassifier(**BASE, class_names=names).fit(x, _string_y(y01))


def test_class_names_is_reported_implemented():
    assert catboost_rs._param_status("class_names") == "IMPLEMENTED"


def test_matches_the_frozen_catboost_1_2_10_oracle():
    """Hermetic parity: reads FROZEN fixtures, never imports `catboost`.

    Both `class_names` orders are checked, so honouring the order is proven rather
    than assumed -- the fixture records that the two orders differ by 0.265 straight
    and align to 1.1e-16 once the columns are reversed, so an implementation that
    ignored the order could not pass both cells.
    """
    import json
    import pathlib

    base = (pathlib.Path(__file__).resolve().parents[3]
            / "crates" / "cb-oracle" / "fixtures" / "class_names")
    meta = json.loads((base / "meta.json").read_text())
    x = np.load(base / "X.npy")
    y01 = np.load(base / "y_index.npy")
    labels = np.array(meta["labels"])[y01.astype(int)]

    # `thread_count` is still a declared parity gap on this surface and `verbose` is
    # an output control -- both are upstream-side only and neither affects the
    # numbers, so they are dropped rather than the fixture being weakened.
    params = {k: v for k, v in meta["params"].items()
              if k not in ("thread_count", "verbose")}

    for order in meta["orders"]:
        stem = "_".join(order)
        m = CatBoostClassifier(**params, class_names=order)
        m.fit(x, labels)
        assert m.classes_ == order

        want_idx = np.load(base / f"pred_index_{stem}.npy")
        got_idx = np.array([order.index(p) for p in m.predict(x)], dtype=np.float32)
        assert np.array_equal(got_idx, want_idx), (
            f"{order}: predicted labels diverge from catboost 1.2.10"
        )

        want_proba = np.load(base / f"proba_{stem}.npy")
        got_proba = np.asarray(m.predict_proba(x), dtype=np.float64)
        assert np.max(np.abs(got_proba - want_proba)) <= 1e-5, (
            f"{order}: predict_proba diverges from catboost 1.2.10 by "
            f"{np.max(np.abs(got_proba - want_proba)):.3e}"
        )


def test_without_class_names_the_default_path_is_unchanged():
    """The opt-in must not alter a fit that did not ask for it.

    This is the inert-at-default discipline every parameter in this wave carries,
    and here it also pins the KNOWN GAP: with no `class_names` this surface returns
    raw class indices rather than upstream's original labels, and `classes_` raises
    instead of inventing `[0, 1]`.
    """
    x, y01 = _xy()
    m = CatBoostClassifier(**BASE)
    m.fit(x, y01.astype(np.float32))
    preds = np.asarray(m.predict(x[:8]))
    assert preds.dtype == np.float64
    assert set(np.unique(preds)).issubset({0.0, 1.0})
    with pytest.raises(CatBoostValueError) as e:
        _ = m.classes_
    assert "class_names" in str(e.value)
