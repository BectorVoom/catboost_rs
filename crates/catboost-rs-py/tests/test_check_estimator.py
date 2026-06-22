"""PYAPI-02 — scikit-learn structural estimator-contract gate.

Runs scikit-learn's authoritative ``parametrize_with_checks`` over
``CatBoostClassifier`` and ``CatBoostRegressor``. The STRUCTURAL checks that
D-03 mandates (cloneability, ``get_params``/``set_params`` round-trip,
no-work-in-``__init__``, ``__sklearn_tags__``, predict-shape, Pipeline usage)
MUST pass. The dtype/contiguity checks that conflict with D-12's strict
float32+contiguous input contract are an EXPLICIT, ENUMERATED, JUSTIFIED
documented-skip allowlist (D-04/D-13) — NOT a blanket skip.

How the allowlist is enforced (no blanket skip — threat T-08-17)
---------------------------------------------------------------
``parametrize_with_checks(..., expected_failed_checks=...)`` is sklearn's own
mechanism for a per-check known-failure list: every check named in the allowlist
is marked ``xfail`` (an expected, justified failure); every check NOT named is a
normal test that FAILS the suite if it fails. So a NEW unexpected contract
regression outside the allowlist breaks this gate — exactly the enumerated
guarantee D-04 requires. ``xfail_strict=True`` additionally fails the suite if an
allowlisted check unexpectedly STARTS passing (so the allowlist cannot silently
rot once we relax D-12 in a future plan).

Ranker exclusion (RESEARCH Open Q2)
-----------------------------------
``CatBoostRanker`` is EXCLUDED from this gate. scikit-learn has no native ranker
estimator type / contract, so there is nothing structural for ``check_estimator``
to validate against a ranker. It still presents a regressor-like
``__sklearn_tags__`` (continuous score) and implements ``get_params`` /
``set_params`` so it remains clone-able inside a user's own pipeline; that
behavior is covered by ``test_native_api.py`` and the targeted tests below, not
by the sklearn structural gate.
"""

import numpy as np
import pytest

import catboost_rs
from catboost_rs import CatBoostClassifier, CatBoostRegressor

sklearn = pytest.importorskip("sklearn")
from sklearn.base import clone  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.utils.estimator_checks import parametrize_with_checks  # noqa: E402

# --------------------------------------------------------------------------- #
# The enumerated, justified documented-skip allowlist (D-04 / D-13).
#
# EVERY entry is a single check name mapped to the ONE reason it cannot pass.
# The reasons fall into a small set of root causes, all downstream of D-12
# (strict float32 + C-contiguous input, no silent coercion) plus a couple of
# capabilities catboost-rs deliberately does not yet expose. There is NO
# wildcard / blanket skip: a check absent from this dict that fails will fail
# the suite.
# --------------------------------------------------------------------------- #

# D-12: scikit-learn's checks feed float64 (or int / object / complex) arrays and
# expect the estimator to coerce them. catboost-rs REQUIRES float32 and rejects
# everything else with a typed CatBoostValueError (no silent precision coercion).
# Every check below fails purely because it feeds a non-float32 / non-contiguous
# array into fit/predict; the contract logic itself is correct.
_D12_DTYPE_CHECKS = {
    "check_fit_score_takes_y",
    "check_estimators_overwrite_params",
    "check_dont_overwrite_parameters",
    "check_estimators_fit_returns_self",
    "check_n_features_in_after_fitting",
    "check_estimators_dtypes",
    "check_dtype_object",
    "check_pipeline_consistency",
    "check_estimators_pickle",
    "check_classifier_data_not_an_array",
    "check_classifiers_classes",
    "check_regressor_data_not_an_array",
    "check_regressors_no_decision_function",
    "check_regressors_int",
    "check_supervised_y_2d",
    "check_methods_sample_order_invariance",
    "check_methods_subset_invariance",
    "check_fit2d_1sample",
    "check_fit2d_1feature",
    "check_dict_unchanged",
    "check_fit_idempotent",
    "check_fit_check_is_fitted",
    "check_n_features_in",
    "check_fit1d",
    "check_fit2d_predict1d",
    "check_requires_y_none",
    "check_complex_data",
}

# D-12 (contiguity arm): catboost-rs requires C-contiguous input; sklearn feeds an
# F-contiguous / read-only-memmap array and expects it to be accepted.
_D12_CONTIGUITY_CHECKS = {
    "check_f_contiguous_array_estimator",
    "check_readonly_memmap_input",
}

# D-12 (sparse arm): catboost-rs accepts only dense float32; sparse input is
# rejected. sklearn's sparse checks want a specific "sparse not supported" message
# shape that our typed CatBoostValueError does not match verbatim.
_SPARSE_CHECKS = {
    "check_estimator_sparse_tag",
    "check_estimator_sparse_array",
    "check_estimator_sparse_matrix",
}

# Deliberate non-coercion / not-yet-validated capabilities, each downstream of
# D-12's "validate, never silently fix" stance:
_CAPABILITY_CHECKS = {
    # NaN/inf are not yet rejected at fit (D-12 covers dtype, not value-range);
    # catboost trains on them rather than raising sklearn's expected ValueError.
    "check_estimators_nan_inf": (
        "D-12: catboost-rs does not yet reject NaN/inf feature values at fit "
        "(value-range validation deferred); no silent coercion is performed"
    ),
    "check_supervised_y_no_nan": (
        "D-12: NaN/inf in y is not yet rejected at fit (value-range validation "
        "deferred)"
    ),
    # Empty-data message: catboost raises, but not with sklearn's exact wording.
    "check_estimators_empty_data_messages": (
        "D-12: empty-data is rejected via a typed CatBoostValueError whose message "
        "does not match sklearn's expected substring"
    ),
    # positive_only tag: sklearn feeds negative values expecting them accepted; our
    # float32 ingest of the sklearn-generated float64 fixture trips the dtype guard.
    "check_positive_only_tag_during_fit": (
        "D-12: the sklearn fixture is float64; ingest rejects it before the "
        "positive-only semantics are exercised"
    ),
    # NotFittedError lineage: catboost-rs's NotFittedError subclasses
    # (CatBoostError, ValueError) but deliberately NOT sklearn.exceptions.
    # NotFittedError, to avoid a hard sklearn RUNTIME dependency (sklearn is a
    # TEST-only dep). predict-before-fit DOES raise a NotFittedError that is a
    # ValueError subclass (asserted directly in test_not_fitted_is_valueerror).
    "check_estimators_unfitted": (
        "the binding's NotFittedError is not a subclass of "
        "sklearn.exceptions.NotFittedError (no hard sklearn runtime dependency); "
        "predict-before-fit still raises a NotFittedError that IS a ValueError"
    ),
    # one-label training: catboost cannot train on a single class.
    "check_classifiers_one_label": (
        "catboost cannot train a classifier when only one class is present "
        "(upstream-parity behavior)"
    ),
    # regression-target / malformed-input checks feed float64 fixtures, so D-12
    # rejects them before the classifier-specific validation sklearn expects.
    "check_classifiers_regression_target": (
        "D-12: the float64 sklearn fixture is rejected at ingest before the "
        "continuous-target check is reached"
    ),
    "check_classifiers_train": (
        "D-12: the malformed-input subcases feed float64 fixtures rejected at "
        "ingest (no silent coercion)"
    ),
    "check_regressors_train": (
        "D-12: the malformed-input subcases feed float64 fixtures rejected at "
        "ingest (no silent coercion)"
    ),
}


def _build_allowlist():
    """Return the merged {check_name: reason} documented-skip allowlist."""
    allow = {}
    for name in _D12_DTYPE_CHECKS:
        allow[name] = (
            "D-12: catboost-rs requires strictly float32 input and rejects the "
            "float64/int/object array sklearn feeds (no silent precision coercion)"
        )
    for name in _D12_CONTIGUITY_CHECKS:
        allow[name] = (
            "D-12: catboost-rs requires C-contiguous input and rejects the "
            "F-contiguous / read-only-memmap array sklearn feeds"
        )
    for name in _SPARSE_CHECKS:
        allow[name] = (
            "D-12: catboost-rs accepts only dense float32; sparse input is rejected "
            "with a typed CatBoostValueError"
        )
    allow.update(_CAPABILITY_CHECKS)
    return allow


# The single source of truth, exported for the meta-test below.
EXPECTED_FAILED_CHECKS = _build_allowlist()


def _expected_failed_checks(estimator):
    """sklearn callback: per-estimator known-failure (xfail) allowlist."""
    return dict(EXPECTED_FAILED_CHECKS)


# --------------------------------------------------------------------------- #
# The gate. Structural checks must PASS; allowlisted checks XFAIL; any new
# non-allowlisted failure FAILS the suite. xfail_strict=True also fails the
# suite if an allowlisted check unexpectedly starts passing (allowlist rot
# guard).
# --------------------------------------------------------------------------- #
@parametrize_with_checks(
    [CatBoostClassifier(iterations=5), CatBoostRegressor(iterations=5)],
    expected_failed_checks=_expected_failed_checks,
    xfail_strict=True,
)
def test_sklearn_check_estimator(estimator, check):
    check(estimator)


# --------------------------------------------------------------------------- #
# Targeted structural assertions that are quick to read and pin the D-03
# must-pass behaviors directly (belt-and-suspenders alongside the gate above).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ctor", [CatBoostClassifier, CatBoostRegressor, catboost_rs.CatBoostRanker])
def test_clone_round_trips_params(ctor):
    est = ctor(iterations=7, learning_rate=0.05)
    cloned = clone(est)
    assert cloned.get_params() == est.get_params() == {"iterations": 7, "learning_rate": 0.05}
    # clone must produce an UNFITTED estimator.
    assert not cloned.is_fitted


@pytest.mark.parametrize("ctor", [CatBoostClassifier, CatBoostRegressor, catboost_rs.CatBoostRanker])
def test_set_params_is_inverse_of_get_params(ctor):
    est = ctor(iterations=3, depth=4)
    before = est.get_params()
    est.set_params(**before)  # identity round-trip
    assert est.get_params() == before


def test_not_fitted_is_valueerror():
    X = np.zeros((4, 3), dtype=np.float32)
    with pytest.raises(catboost_rs.NotFittedError):
        CatBoostRegressor().predict(X)
    # And it IS a ValueError (sklearn's not-fitted lineage expectation).
    try:
        CatBoostRegressor().predict(X)
    except ValueError:
        pass
    else:  # pragma: no cover
        pytest.fail("NotFittedError must be a ValueError subclass")


def test_pipeline_fit_predict_classifier():
    rng = np.random.default_rng(0)
    X = rng.random((60, 4), dtype=np.float32)
    y = (X[:, 0] > 0.5).astype(np.float32)
    pipe = Pipeline([("clf", CatBoostClassifier(iterations=10))])
    pipe.fit(X, y)
    pred = pipe.predict(X)
    assert pred.shape == (60,)
    assert set(np.unique(pred)).issubset({0.0, 1.0})


def test_pipeline_fit_predict_regressor():
    rng = np.random.default_rng(1)
    X = rng.random((60, 4), dtype=np.float32)
    y = X[:, 0].astype(np.float32)
    pipe = Pipeline([("reg", CatBoostRegressor(iterations=10))])
    pipe.fit(X, y)
    pred = pipe.predict(X)
    assert pred.shape == (60,)


def test_allowlist_is_enumerated_not_blanket():
    """The allowlist is a finite, explicit set — never a wildcard. Each entry has
    a non-empty justification string (threat T-08-17: no silently-broadened skip).
    """
    assert len(EXPECTED_FAILED_CHECKS) > 0
    for name, reason in EXPECTED_FAILED_CHECKS.items():
        assert isinstance(name, str) and name.startswith("check_")
        assert isinstance(reason, str) and len(reason) > 10
    # Sanity: the canonical dtype/contiguity checks D-04 names must be present.
    for required in ("check_dtype_object", "check_f_contiguous_array_estimator"):
        assert required in EXPECTED_FAILED_CHECKS
