"""CM-PY Python-surface test: `save_coreml` on `CatBoostRegressor` (EXPORT-02).

Mirrors `test_onnx_export.py`'s unfitted-guard / fitted-write shape.

Two distinct capabilities, deliberately split into two tests, because they have
DIFFERENT availability:

* **Decoding** an `.mlmodel` through coremltools' own protobuf loader works on
  ANY platform — it is pure protobuf. This is a REAL structural oracle on
  Linux, not a caveat, and it is asserted in full below.
* **Executing** one, `MLModel.predict(...)`, requires Apple's native runtime
  (`coremltools.libcoremlpython`), which ships only on macOS. On Linux
  coremltools raises `Exception: Model prediction is only supported on macOS
  version 10.13 or later`.

These used to live in ONE test, so the whole structural check was lost to the
macOS-only `predict` call on every non-Apple host.
"""

import platform

import pytest


def _coreml_runtime_available(model) -> bool:
    """Whether Apple's native CoreML runtime can actually execute `model`.

    coremltools loads a `__proxy__` only when `libcoremlpython` is importable,
    which is macOS-only; without it `predict()` raises. Check the proxy rather
    than the platform string alone, so a macOS host with a broken/partial
    install also skips cleanly instead of erroring.
    """
    return platform.system() == "Darwin" and getattr(model, "__proxy__", None) is not None


def test_unfitted_regressor_save_coreml_raises_not_fitted(tmp_path):
    """An unfitted CatBoostRegressor.save_coreml raises NotFittedError."""
    import catboost_rs

    model = catboost_rs.CatBoostRegressor(iterations=5)
    path = str(tmp_path / "unfitted.mlmodel")
    raised = False
    try:
        model.save_coreml(path)
    except catboost_rs.NotFittedError:
        raised = True
    assert raised


def test_fitted_regressor_save_coreml_writes_nonempty_file(tmp_path, toy_regression):
    """A fitted CatBoostRegressor.save_coreml succeeds; the file exists and is
    non-empty."""
    import catboost_rs

    x, y = toy_regression
    model = catboost_rs.CatBoostRegressor(iterations=10, depth=3).fit(x, y)
    path = tmp_path / "regressor.mlmodel"
    model.save_coreml(str(path))

    assert path.exists()
    assert len(path.read_bytes()) > 0


def _fit_and_export(tmp_path, toy_regression, name, iterations=10, depth=3):
    import catboost_rs

    x, y = toy_regression
    model = catboost_rs.CatBoostRegressor(iterations=iterations, depth=depth).fit(x, y)
    path = tmp_path / name
    model.save_coreml(str(path))
    return x, path


def test_fitted_regressor_save_coreml_decodes_via_coremltools(tmp_path, toy_regression):
    """The written `.mlmodel` round-trips through coremltools' protobuf loader
    and carries the structure it claims to.

    Platform-independent: decoding is pure protobuf. This is the real
    structural oracle for the exporter — it is what would catch a wrong input
    arity, a dropped tree, or a mis-declared output.
    """
    coremltools = pytest.importorskip(
        "coremltools",
        reason="coremltools is not installed in this environment; install it "
        "(e.g. `uv pip install --python .venv/bin/python coremltools`) to run "
        "the CoreML decode round-trip test",
    )

    iterations = 10
    x, path = _fit_and_export(
        tmp_path, toy_regression, "regressor_rt.mlmodel", iterations=iterations
    )
    spec = coremltools.models.MLModel(str(path)).get_spec()

    # A plain tree-ensemble regressor, not a pipeline.
    assert spec.WhichOneof("Type") == "treeEnsembleRegressor"

    # One SCALAR double input per float feature, in feature order. (The old
    # single-array `predict` call in this test contradicted exactly this.)
    n_features = x.shape[1]
    assert [i.name for i in spec.description.input] == [
        f"feature_{i}" for i in range(n_features)
    ]
    assert {i.type.WhichOneof("Type") for i in spec.description.input} == {"doubleType"}

    # A single named output, which is also the declared prediction.
    assert [o.name for o in spec.description.output] == ["prediction"]
    assert spec.description.predictedFeatureName == "prediction"

    # Every boosting iteration reached the ensemble — a dropped tree is the
    # failure mode a "file is non-empty" check cannot see.
    nodes = spec.treeEnsembleRegressor.treeEnsemble.nodes
    assert nodes, "the exported ensemble carries no nodes"
    assert len({n.treeId for n in nodes}) == iterations


@pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="MLModel.predict requires Apple's native CoreML runtime "
    "(coremltools.libcoremlpython), which ships only on macOS; coremltools "
    "raises 'Model prediction is only supported on macOS version 10.13 or "
    "later' elsewhere. The platform-independent structural check lives in "
    "test_fitted_regressor_save_coreml_decodes_via_coremltools.",
)
def test_fitted_regressor_coreml_predicts_finite_values(tmp_path, toy_regression):
    """macOS only: the exported model actually EXECUTES and yields finite values."""
    coremltools = pytest.importorskip("coremltools")
    import numpy as np

    x, path = _fit_and_export(tmp_path, toy_regression, "regressor_exec.mlmodel")
    loaded = coremltools.models.MLModel(str(path))
    if not _coreml_runtime_available(loaded):
        pytest.skip("coremltools is installed without its native runtime proxy")

    # ONE SCALAR PER INPUT. The previous version passed the whole 4-feature row
    # under `feature_0`, which the spec's four scalar doubleType inputs would
    # have rejected even on macOS.
    row = np.asarray(x[0], dtype=np.float64)
    out = loaded.predict(
        {i.name: float(row[k]) for k, i in enumerate(loaded.get_spec().description.input)}
    )
    prediction = np.asarray(next(iter(out.values())))
    assert np.all(np.isfinite(prediction))
