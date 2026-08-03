"""PARAM-02 — the fit-parameter surface promoted from KNOWN_NOT_YET to IMPLEMENTED.

Seventeen upstream params became reachable once `CatBoostBuilder` grew a setter
for each (PARAM-01), together with the `eval_set` fit kwarg the four
detector/best-model params need in order to do anything at all.

These tests assert the params CHANGE THE FIT, not merely that they are accepted.
A parameter that parses and is then dropped would pass an "it doesn't raise"
test while training exactly the model the user did not ask for — which is the
failure the registry's honesty policy exists to prevent. So each test compares
against a control run and asserts the predictions (or the accepted/rejected
status) actually differ.

The eval sets here are deliberately ANTI-CORRELATED with the learn set (same
features, negated target). That makes the validation metric worsen from the very
first iteration, so early stopping and best-model truncation have an exact,
non-statistical effect rather than one that depends on the split.
"""

import numpy as np
import pytest

from catboost_rs import (
    CatBoostClassifier,
    CatBoostParameterError,
    CatBoostRegressor,
    Pool,
)


def _xy(n=200, k=4, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random((n, k), dtype=np.float32)
    y = (x[:, 0] * 3.0 - x[:, 1] * 2.0).astype(np.float32)
    return x, y


def _anticorrelated(x, y):
    """An eval set whose target is the NEGATION of the learn relationship."""
    return x, (-y).astype(np.float32)


# ─── eval_set plumbing ───────────────────────────────────────────────────────


@pytest.mark.parametrize("shape", ["tuple", "list_of_tuples", "pool", "list_of_pools"])
def test_eval_set_accepts_every_upstream_shape(shape):
    x, y = _xy()
    xv, yv = _anticorrelated(*_xy(n=60, seed=1))
    eval_set = {
        "tuple": (xv, yv),
        "list_of_tuples": [(xv, yv)],
        "pool": Pool(xv, yv),
        "list_of_pools": [Pool(xv, yv)],
    }[shape]
    CatBoostRegressor(iterations=10, depth=3).fit(x, y, eval_set=eval_set)


def test_eval_set_of_a_wrong_width_is_rejected():
    x, y = _xy(k=4)
    xv, yv = _xy(n=50, k=2, seed=1)
    with pytest.raises(Exception) as excinfo:
        CatBoostRegressor(iterations=5).fit(x, y, eval_set=(xv, yv))
    msg = str(excinfo.value)
    assert "eval set" in msg and "float features" in msg, msg


def test_a_malformed_eval_set_is_rejected():
    x, y = _xy()
    with pytest.raises(Exception):
        CatBoostRegressor(iterations=5).fit(x, y, eval_set=(x, y, x))
    with pytest.raises(Exception):
        CatBoostRegressor(iterations=5).fit(x, y, eval_set=42)


def test_eval_set_none_matches_a_plain_fit_exactly():
    """The eval-set plumbing must not perturb a learn-only fit."""
    x, y = _xy()
    a = CatBoostRegressor(iterations=15, depth=3, learning_rate=0.2)
    a.fit(x, y)
    b = CatBoostRegressor(iterations=15, depth=3, learning_rate=0.2)
    b.fit(x, y, eval_set=None)
    np.testing.assert_array_equal(a.predict(x), b.predict(x))


# ─── the eval-set-only params ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs",
    [
        {"od_type": "Iter", "od_wait": 3},
        {"od_pval": 0.01},
        {"early_stopping_rounds": 3},
        {"use_best_model": True},
    ],
)
def test_eval_set_only_params_raise_without_an_eval_set(kwargs):
    """Silently inert is not acceptable: it must raise instead."""
    x, y = _xy()
    with pytest.raises(CatBoostParameterError) as excinfo:
        CatBoostRegressor(iterations=10, **kwargs).fit(x, y)
    msg = str(excinfo.value)
    assert "eval_set" in msg, msg


def test_early_stopping_rounds_produces_a_smaller_model():
    """Early stopping must actually stop — observable as a shorter ensemble.

    Compared against the SAME configuration without the detector, which is what
    rules out "the run was short for some unrelated reason".
    """
    x, y = _xy()
    xv, yv = _anticorrelated(*_xy(n=60, seed=1))

    stopped = CatBoostRegressor(
        iterations=200, depth=3, learning_rate=0.2, early_stopping_rounds=2
    )
    stopped.fit(x, y, eval_set=(xv, yv))
    full = CatBoostRegressor(iterations=200, depth=3, learning_rate=0.2)
    full.fit(x, y)

    # The two models cannot both be the same ensemble: the stopped run grew far
    # fewer trees, so its predictions on the learn set are measurably less fitted.
    stopped_err = float(np.mean((stopped.predict(x) - y) ** 2))
    full_err = float(np.mean((full.predict(x) - y) ** 2))
    assert stopped_err > full_err, (
        f"early stopping must yield a less-fitted model; got stopped={stopped_err} "
        f"full={full_err}"
    )


def test_use_best_model_truncates_to_the_best_iteration():
    """On an anti-correlated eval set the best iteration is the first, so the
    truncated model is very nearly the bias alone."""
    x, y = _xy()
    xv, yv = _anticorrelated(*_xy(n=60, seed=1))

    truncated = CatBoostRegressor(
        iterations=40, depth=3, learning_rate=0.2, use_best_model=True
    )
    truncated.fit(x, y, eval_set=(xv, yv))
    full = CatBoostRegressor(iterations=40, depth=3, learning_rate=0.2)
    full.fit(x, y, eval_set=(xv, yv))

    truncated_err = float(np.mean((truncated.predict(x) - y) ** 2))
    full_err = float(np.mean((full.predict(x) - y) ** 2))
    assert truncated_err > full_err, (
        "use_best_model must drop the trees grown after the best iteration"
    )


def test_early_stopping_rounds_conflicts_with_explicit_od_params():
    x, y = _xy()
    xv, yv = _anticorrelated(*_xy(n=60, seed=1))
    with pytest.raises(CatBoostParameterError) as excinfo:
        CatBoostRegressor(
            iterations=20, early_stopping_rounds=3, od_wait=5
        ).fit(x, y, eval_set=(xv, yv))
    msg = str(excinfo.value)
    assert "early_stopping_rounds" in msg and "od_wait" in msg, msg


def test_eval_metric_accepts_a_parametric_descriptor():
    x, y = _xy()
    xv, yv = _anticorrelated(*_xy(n=60, seed=1))
    CatBoostRegressor(iterations=10, eval_metric="Quantile:alpha=0.9").fit(
        x, y, eval_set=(xv, yv)
    )


def test_an_unsupported_eval_metric_is_rejected_by_name():
    x, y = _xy()
    with pytest.raises(CatBoostParameterError) as excinfo:
        CatBoostRegressor(iterations=5, eval_metric="NotAMetric").fit(x, y)
    assert "NotAMetric" in str(excinfo.value)


# ─── the always-on promoted params ───────────────────────────────────────────


# Each case is (control kwargs, tuned kwargs). Most params are compared against
# a bare default fit — but three are only CONSUMED in a particular regime, so
# comparing them against the default would assert something the engine (and
# upstream) never promised:
#
#   * `min_data_in_leaf` is read by the LEAF-WISE growers only; under the default
#     SymmetricTree policy the structure is bounded by `depth` and the value is
#     inert. Paired with `grow_policy="Lossguide"`.
#   * `boosting_type` is compared as Plain-vs-ORDERED; "Plain" IS the default, so
#     asserting it changes anything would be asserting a bug.
#
# `has_time` is absent from this list entirely — see
# `test_has_time_is_observable_only_with_categorical_features` below for why, and
# for where it IS observable.
PARAM_CASES = [
    ({}, {"grow_policy": "Lossguide", "max_leaves": 8}),
    ({}, {"grow_policy": "Depthwise"}),
    ({}, {"feature_weights": [5.0, 0.1, 0.1, 0.1]}),
    ({}, {"first_feature_use_penalties": [10.0, 0.0, 0.0, 0.0]}),
    ({}, {"per_object_feature_penalties": [1.0, 0.0, 0.0, 0.0]}),
    ({}, {"monotone_constraints": [1, 0, 0, 0]}),
    ({}, {"boosting_type": "Ordered"}),
    (
        {"grow_policy": "Lossguide"},
        {"grow_policy": "Lossguide", "min_data_in_leaf": 20},
    ),
]


@pytest.mark.parametrize("control,tuned_kwargs", PARAM_CASES)
def test_promoted_params_change_the_trained_model(control, tuned_kwargs):
    """Each promoted param must MOVE the predictions away from its control fit.

    An accepted-but-dropped parameter is the exact failure mode the honesty
    policy targets, and it is invisible to a test that only checks `fit` returns.
    """
    x, y = _xy()
    base = CatBoostRegressor(iterations=20, depth=3, learning_rate=0.2, **control)
    base.fit(x, y)
    tuned = CatBoostRegressor(
        iterations=20, depth=3, learning_rate=0.2, **tuned_kwargs
    )
    tuned.fit(x, y)

    assert not np.allclose(base.predict(x), tuned.predict(x)), (
        f"{tuned_kwargs} was accepted but did not change the model"
    )


def test_min_data_in_leaf_is_inert_under_the_symmetric_grower():
    """The documented flip side of the pairing above, asserted rather than assumed.

    `min_data_in_leaf` is read by the LEAF-WISE growers only; the default
    SymmetricTree structure is bounded by `depth`, so the value cannot bite.
    Pinning that here is what stops the paired case above from silently masking a
    regression: if the param ever started perturbing the symmetric path, this test
    would fail rather than both quietly passing.
    """
    x, y = _xy()
    base = CatBoostRegressor(iterations=20, depth=3, learning_rate=0.2)
    base.fit(x, y)
    tuned = CatBoostRegressor(
        iterations=20, depth=3, learning_rate=0.2, min_data_in_leaf=20
    )
    tuned.fit(x, y)
    np.testing.assert_array_equal(base.predict(x), tuned.predict(x))


def test_has_time_is_observable_only_with_categorical_features():
    """`has_time` is a CTR-path parameter in this engine, and the test says so.

    `need_shuffle` (boosting.rs:616) is
    `(has_cat_features || Ordered) && !has_time`, but its result is consumed ONLY
    inside the CTR branch of `train_inner` (boosting.rs:3634, guarded by
    `cat_learn_permutation.is_some()`, which requires CTR candidates). So on a
    FLOAT-ONLY pool `has_time` cannot change the model — not even under Ordered
    boosting, because the ordered approximant path does not read the learn-set
    shuffle. Both halves are asserted:

      * float-only + Ordered -> byte-identical (the honest inert case);
      * a CTR-routed categorical column -> the model MOVES.

    Recording the float-only half as an equality rather than omitting it is what
    keeps the limitation visible: if the shuffle is ever threaded into the ordered
    numeric path, this test fails loudly instead of the parameter silently
    changing meaning.
    """
    x, y = _xy()
    plain = CatBoostRegressor(iterations=15, depth=3, boosting_type="Ordered")
    plain.fit(x, y)
    timed = CatBoostRegressor(
        iterations=15, depth=3, boosting_type="Ordered", has_time=True
    )
    timed.fit(x, y)
    np.testing.assert_array_equal(
        plain.predict(x),
        timed.predict(x),
        err_msg="has_time must be inert on a float-only fit (no CTR shuffle to skip)",
    )

    # The regime where it IS consumed: a high-cardinality categorical column
    # routes to the CTR path, whose materialization order carries the shuffle.
    #
    # The TARGET must depend on the categorical level. A cat column whose level
    # carries no signal is present in the pool but never chosen as a split, so no
    # CTR column is ever materialized and every CTR-path parameter — `has_time`,
    # `simple_ctr`, `max_ctr_complexity` — is equally inert. Deriving `yc` from
    # the level is what puts the CTR path on the critical path of the fit.
    pd = pytest.importorskip("pandas")
    n = 200
    level = np.array([i % 25 for i in range(n)])
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "f0": rng.random(n).astype(np.float32),
            "f1": rng.random(n).astype(np.float32),
            "c": [f"lvl{v}" for v in level],
        }
    )
    yc = (level * 0.5).astype(np.float32)

    shuffled = CatBoostRegressor(iterations=15, depth=3)
    shuffled.fit(df, yc, cat_features=[2])
    ordered = CatBoostRegressor(iterations=15, depth=3, has_time=True)
    ordered.fit(df, yc, cat_features=[2])

    assert not np.allclose(shuffled.predict(df), ordered.predict(df)), (
        "has_time must change a CTR-routed fit: it skips the initial learn-set "
        "shuffle the CTR materialization order carries"
    )


@pytest.mark.parametrize(
    "spelling",
    [[1, 0, 0, 0], "1,0,0,0", "(1,0,0,0)", {0: 1}],
)
def test_monotone_constraint_spellings_agree(spelling):
    """All four upstream spellings must train the SAME model."""
    x, y = _xy()
    reference = CatBoostRegressor(
        iterations=15, depth=3, monotone_constraints=[1, 0, 0, 0]
    )
    reference.fit(x, y)
    other = CatBoostRegressor(iterations=15, depth=3, monotone_constraints=spelling)
    other.fit(x, y)
    np.testing.assert_array_equal(reference.predict(x), other.predict(x))


def test_a_name_keyed_monotone_dict_is_rejected():
    """The engine applies constraints positionally and a Pool carries no feature
    names, so a named dict cannot be honoured — and must not be dropped."""
    x, y = _xy()
    with pytest.raises(CatBoostParameterError) as excinfo:
        CatBoostRegressor(iterations=5, monotone_constraints={"f0": 1}).fit(x, y)
    assert "feature NAME" in str(excinfo.value)


def test_region_grow_policy_is_rejected_as_a_parity_gap():
    x, y = _xy()
    with pytest.raises(CatBoostParameterError) as excinfo:
        CatBoostRegressor(iterations=5, grow_policy="Region").fit(x, y)
    msg = str(excinfo.value)
    assert "Region" in msg and "parity gap" in msg, msg


@pytest.mark.parametrize(
    "kwargs",
    [
        {"od_pval": 1.5},
        {"max_leaves": 1},
        {"fold_len_multiplier": 1.0},
        {"early_stopping_rounds": 0},
        {"penalties_coefficient": -1.0},
    ],
)
def test_out_of_range_values_are_rejected(kwargs):
    x, y = _xy()
    xv, yv = _anticorrelated(*_xy(n=60, seed=1))
    with pytest.raises(CatBoostParameterError):
        CatBoostRegressor(iterations=10, **kwargs).fit(x, y, eval_set=(xv, yv))


def test_lightgbm_leaf_aliases_match_their_canonical_names():
    x, y = _xy()
    canonical = CatBoostRegressor(
        iterations=15, depth=3, grow_policy="Lossguide", max_leaves=8,
        min_data_in_leaf=5,
    )
    canonical.fit(x, y)
    aliased = CatBoostRegressor(
        iterations=15, depth=3, grow_policy="Lossguide", num_leaves=8,
        min_child_samples=5,
    )
    aliased.fit(x, y)
    np.testing.assert_array_equal(canonical.predict(x), aliased.predict(x))


def test_the_classifier_takes_the_same_surface():
    """The promoted params are wired on the classifier and ranker too, not just
    the regressor — three separate `fit` implementations, three chances to miss
    one."""
    x, _ = _xy()
    y = (x[:, 0] > 0.5).astype(np.float32)
    xv = x[:40]
    yv = (1.0 - y[:40]).astype(np.float32)

    clf = CatBoostClassifier(
        iterations=30, depth=3, learning_rate=0.2, early_stopping_rounds=2
    )
    clf.fit(x, y, eval_set=(xv, yv))
    assert clf.predict(x).shape == (x.shape[0],)

    with pytest.raises(CatBoostParameterError):
        CatBoostClassifier(iterations=10, use_best_model=True).fit(x, y)
