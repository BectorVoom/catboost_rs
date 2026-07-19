"""EXPORT-01f Python-surface tests: `save_onnx` on `CatBoostRegressor` /
`CatBoostClassifier` (AT-01f-2, AT-01f-3, AT-01f-4).

AT-01f-4's "assert the exported graph is a classifier, not a regressor" check
uses a RAW BYTE search over the written `.onnx` file rather than a protobuf
parser: `onnx`/`protobuf` are not part of this project's pinned test
dependencies (`README.md`'s `maturin develop` venv recipe), and the plan's own
guidance permits resolving this choice at implementation time. `op_type` is
stored as a length-delimited UTF-8 string field in the serialized
`NodeProto`, so the literal ASCII operator name appears verbatim in the file
bytes — a lightweight, dependency-free structural check that still
distinguishes `TreeEnsembleClassifier`+`ZipMap` from `TreeEnsembleRegressor`.

# ONNX-runtime round-trip tests (code-review Fix 3)

Every test above (and every existing Rust `onnx.rs`/`onnx_test.rs` test) only
checks STRUCTURAL properties or self-consistency against the exporter's own
formulas — nothing independently verifies that the reversed-split-order tree
walk or the dimension-major multiclass leaf-indexing (both flagged in
`cb-model/src/export/onnx.rs`'s own module doc as the load-bearing pitfalls of
this feature) actually produce numerically correct predictions once the
exported file is loaded and run by a REAL ONNX consumer. The tests below close
that gap with `onnxruntime`, guarded by `pytest.importorskip` since
`onnxruntime` is not part of this project's pinned test dependencies (like
`numpy`/`pytest`/`scikit-learn`, it is installed ad hoc into `.venv-py8`, e.g.
via `uv pip install --python .venv-py8/bin/python3 onnxruntime` — see
`README.md`'s venv recipe, which likewise never lists `pytest` explicitly).
Float32 tolerance (`atol=1e-4`) reflects `onnx.rs`'s own `attr_floats` doc
comment: ONNX tree-ensemble attributes are single-precision, so agreement is
bounded by float32 rounding, not float64 exactness.
"""

import numpy as np
import pytest

onnxruntime = pytest.importorskip(
    "onnxruntime",
    reason="onnxruntime is not installed in this environment; ad hoc install "
    "via `uv pip install --python .venv-py8/bin/python3 onnxruntime` to run "
    "the ONNX-runtime round-trip tests locally",
)


def test_unfitted_regressor_save_onnx_raises_not_fitted(tmp_path):
    """AT-01f-2: an unfitted CatBoostRegressor.save_onnx raises NotFittedError."""
    import catboost_rs

    model = catboost_rs.CatBoostRegressor(iterations=5)
    path = str(tmp_path / "unfitted.onnx")
    raised = False
    try:
        model.save_onnx(path)
    except catboost_rs.NotFittedError:
        raised = True
    assert raised


def test_unfitted_classifier_save_onnx_raises_not_fitted(tmp_path):
    """AT-01f-2 (classifier arm): same NotFittedError guard."""
    import catboost_rs

    model = catboost_rs.CatBoostClassifier(iterations=5)
    path = str(tmp_path / "unfitted_clf.onnx")
    raised = False
    try:
        model.save_onnx(path)
    except catboost_rs.NotFittedError:
        raised = True
    assert raised


def test_fitted_regressor_save_onnx_writes_nonempty_file(tmp_path, toy_regression):
    """AT-01f-3: a fitted CatBoostRegressor.save_onnx succeeds; file exists,
    is non-empty, and is a TreeEnsembleRegressor (not a classifier) graph.
    """
    import catboost_rs

    x, y = toy_regression
    model = catboost_rs.CatBoostRegressor(iterations=10, depth=3).fit(x, y)
    path = tmp_path / "regressor.onnx"
    model.save_onnx(str(path))

    assert path.exists()
    data = path.read_bytes()
    assert len(data) > 0
    assert b"TreeEnsembleRegressor" in data
    assert b"TreeEnsembleClassifier" not in data


def test_fitted_classifier_save_onnx_emits_classifier_graph(tmp_path, toy_classification):
    """AT-01f-4: a fitted CatBoostClassifier.save_onnx succeeds and the
    exported graph contains a TreeEnsembleClassifier+ZipMap pair, not a
    TreeEnsembleRegressor.
    """
    import catboost_rs

    x, y = toy_classification
    model = catboost_rs.CatBoostClassifier(iterations=10, depth=3).fit(x, y)
    path = tmp_path / "classifier.onnx"
    model.save_onnx(str(path))

    assert path.exists()
    data = path.read_bytes()
    assert len(data) > 0
    assert b"TreeEnsembleClassifier" in data
    assert b"ZipMap" in data
    assert b"TreeEnsembleRegressor" not in data


# --- ONNX-runtime round-trip tests (code-review Fix 3) ----------------------


def test_regressor_onnx_runtime_matches_predict(tmp_path, toy_regression):
    """A REAL ONNX-runtime round trip for the regressor path: exports a fitted
    `CatBoostRegressor`, loads it with `onnxruntime.InferenceSession`, and
    compares the runtime's `predictions` output to the model's own
    `.predict(X)` on the SAME input. This exercises the reversed-split-order
    tree walk (`onnx.rs` EXPORT-01b) end-to-end through a real consumer, not
    just the exporter's own formulas.
    """
    x, y = toy_regression
    import catboost_rs

    model = catboost_rs.CatBoostRegressor(iterations=10, depth=3).fit(x, y)
    path = tmp_path / "regressor_rt.onnx"
    model.save_onnx(str(path))

    session = onnxruntime.InferenceSession(str(path))
    (onnx_preds,) = session.run(None, {"features": x})
    onnx_preds = np.asarray(onnx_preds).reshape(-1)

    py_preds = model.predict(x)
    np.testing.assert_allclose(onnx_preds, py_preds, atol=1e-4, rtol=1e-4)


def test_binary_classifier_onnx_runtime_matches_predict(tmp_path, toy_classification):
    """The same round trip for the BINARY classifier path: compares
    onnxruntime's `label` output to `.predict(X)` (exact match — both are an
    argmax over the same two-class LOGISTIC probability) and its
    `probabilities` (`ZipMap` `seq(map(int64,float))`) output to
    `.predict_proba(X)`.
    """
    x, y = toy_classification
    import catboost_rs

    model = catboost_rs.CatBoostClassifier(iterations=10, depth=3).fit(x, y)
    path = tmp_path / "classifier_rt.onnx"
    model.save_onnx(str(path))

    session = onnxruntime.InferenceSession(str(path))
    onnx_labels, onnx_probs = session.run(None, {"features": x})

    py_labels = model.predict(x)
    py_proba = model.predict_proba(x)

    np.testing.assert_array_equal(np.asarray(onnx_labels), py_labels.astype(np.int64))

    onnx_proba = np.array([[row[0], row[1]] for row in onnx_probs])
    np.testing.assert_allclose(onnx_proba, py_proba, atol=1e-4, rtol=1e-4)


def test_multiclass_classifier_onnx_runtime_matches_upstream_oracle(
    tmp_path, oracle_multiclass_softmax
):
    """The DIMENSION-MAJOR multiclass leaf-indexing pitfall `onnx.rs` EXPORT-01d
    flags in its own module doc cannot be exercised through `.fit()` +
    `.predict()`/`.predict_proba()` on THIS Python surface today: both are
    still hardcoded to the BINARY two-column convention (`classifier.rs`) and
    produce a nonsensical shape for a `dim > 1` model — a separate, pre-existing
    facade gap out of scope for this fix. So this test bypasses `.fit()` /
    `.predict()` entirely: it loads the frozen upstream catboost 1.2.10
    `multiclass_softmax` model via the PUBLIC `load_model` (no training,
    `oracle_multiclass_softmax` fixture), exports it, runs the SAME
    `numeric_tiny` input through onnxruntime, and compares the SOFTMAX
    `probabilities` output to the frozen `predictions.npy` the Rust
    `multiclass_apply_oracle_test.rs` already locks to <= 1e-5 against upstream
    catboost — an INDEPENDENT upstream ground truth, not a self-consistency
    check against this exporter's own formulas. A leaf/class transposition bug
    in the dimension-major read would show up here as a large divergence.
    """
    import catboost_rs

    model = catboost_rs.CatBoostClassifier.load_model(oracle_multiclass_softmax["json"])
    path = tmp_path / "multiclass_rt.onnx"
    model.save_onnx(str(path))

    x = oracle_multiclass_softmax["X"]
    ref = oracle_multiclass_softmax["ref_probabilities"]
    n_classes = ref.shape[1]

    session = onnxruntime.InferenceSession(str(path))
    _onnx_labels, onnx_probs = session.run(None, {"features": x})
    onnx_proba = np.array([[row.get(c, 0.0) for c in range(n_classes)] for row in onnx_probs])

    np.testing.assert_allclose(onnx_proba, ref, atol=1e-4, rtol=1e-4)
