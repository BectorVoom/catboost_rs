"""Python-surface oracle parity vs catboost 1.2.10 (08-04, the phase bar).

The single deterministic oracle path (Path (a), RESEARCH Open Q3 RESOLVED): load
the EXISTING offline catboost 1.2.10 reference `.cbm` / `.json` model via the
`load_model(path)` classmethod, predict on the same `numeric_tiny` input matrix
the Rust `cb-oracle` harness uses, and assert the Python predictions reproduce the
stored reference vector to within 1e-5 (`atol=1e-5, rtol=0`).

Hermetic: reads ONLY frozen fixture files
(`crates/cb-oracle/fixtures/model_serde/{regression,binclf}` + the `numeric_tiny`
input). It NEVER imports the `catboost` package. There is NO re-fit fallback — the
load_model path is the only oracle path.

Both `model_serde` fixtures store `RawFormulaVal` reference vectors:
  * regression  -> compared directly against `CatBoostRegressor.predict`.
  * binclf      -> the raw logit; `CatBoostClassifier.predict_proba[:, 1]` is
                   compared against `sigmoid(raw)` (and `predict` against
                   `raw > 0`), which pins the classifier's probability/label
                   surface numerically to the upstream raw scores.
"""

import numpy as np

import catboost_rs


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def test_regression_oracle_parity_cbm(oracle_regression):
    """`CatBoostRegressor.load_model(.cbm).predict` == catboost 1.2.10 to 1e-5."""
    est = catboost_rs.CatBoostRegressor.load_model(oracle_regression["cbm"])
    py_pred = est.predict(oracle_regression["X"])
    np.testing.assert_allclose(
        py_pred, oracle_regression["ref"], atol=1e-5, rtol=0
    )


def test_regression_oracle_parity_json(oracle_regression):
    """The same parity holds loading the `.json` form of the reference model."""
    est = catboost_rs.CatBoostRegressor.load_model(oracle_regression["json"])
    py_pred = est.predict(oracle_regression["X"])
    np.testing.assert_allclose(
        py_pred, oracle_regression["ref"], atol=1e-5, rtol=0
    )


def test_classification_oracle_parity_proba(oracle_binclf):
    """`CatBoostClassifier.load_model(.cbm).predict_proba[:, 1]` == sigmoid(raw)."""
    est = catboost_rs.CatBoostClassifier.load_model(oracle_binclf["cbm"])
    proba = est.predict_proba(oracle_binclf["X"])
    assert proba.shape == (oracle_binclf["X"].shape[0], 2)
    expected_p1 = _sigmoid(oracle_binclf["ref_raw"])
    np.testing.assert_allclose(proba[:, 1], expected_p1, atol=1e-5, rtol=0)
    # The two-column simplex: class-0 probability is the complement.
    np.testing.assert_allclose(proba[:, 0], 1.0 - expected_p1, atol=1e-5, rtol=0)


def test_classification_oracle_parity_labels(oracle_binclf):
    """`CatBoostClassifier.load_model(.cbm).predict` == (raw > 0) class labels."""
    est = catboost_rs.CatBoostClassifier.load_model(oracle_binclf["cbm"])
    labels = est.predict(oracle_binclf["X"])
    expected = (oracle_binclf["ref_raw"] > 0).astype(np.float64)
    np.testing.assert_allclose(labels, expected, atol=1e-5, rtol=0)
