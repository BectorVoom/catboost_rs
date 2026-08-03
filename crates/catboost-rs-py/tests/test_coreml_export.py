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


def _predict_payload(spec, row):
    """Build the `MLModel.predict` input dict for one feature `row`.

    The spec declares one SCALAR `doubleType` input per float feature, so the
    payload is one scalar per declared input name — NOT the whole row under a
    single key. Extracted from the macOS-only execute test so the marshalling
    itself can be verified on any platform (see
    `test_predict_payload_matches_the_declared_inputs`).
    """
    return {inp.name: float(row[k]) for k, inp in enumerate(spec.description.input)}


def _payload_conformance_errors(payload, spec):
    """Problems that would make `payload` invalid for `spec`, as a list.

    Returns `[]` for a conformant payload. This is the check that makes the
    original defect visible without an Apple runtime.
    """
    problems = []
    declared = [inp.name for inp in spec.description.input]
    if sorted(payload) != sorted(declared):
        problems.append(
            f"payload keys {sorted(payload)} != declared inputs {sorted(declared)}"
        )
    for name, value in payload.items():
        # A `doubleType` input takes ONE number. A list/ndarray under a scalar
        # input is exactly the original bug.
        if hasattr(value, "__len__"):
            problems.append(
                f"input '{name}' is declared scalar doubleType but got a "
                f"sequence of length {len(value)}"
            )
        elif not isinstance(value, (int, float)):
            problems.append(f"input '{name}' is not a number: {type(value).__name__}")
    return problems


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


def test_predict_payload_matches_the_declared_inputs(tmp_path, toy_regression):
    """The `predict` input marshalling is correct — checked WITHOUT a runtime.

    THE ORIGINAL DEFECT: the execute test passed the whole feature ROW as one
    array under `feature_0`:

        loaded.predict({input_name: np.asarray(x[0], dtype=np.float32)})

    but the spec declares one SCALAR `doubleType` input PER feature, so that
    payload was malformed and would have been rejected on macOS too. Because
    the only test using it is macOS-only, the bug was invisible on this host —
    and so was its fix. This test closes that hole: it exercises the real
    `_predict_payload` helper the execute test now calls, on any platform.
    """
    coremltools = pytest.importorskip("coremltools")
    import numpy as np

    x, path = _fit_and_export(tmp_path, toy_regression, "regressor_payload.mlmodel")
    spec = coremltools.models.MLModel(str(path)).get_spec()
    row = np.asarray(x[0], dtype=np.float64)

    payload = _predict_payload(spec, row)
    assert _payload_conformance_errors(payload, spec) == []
    # One entry per feature, positionally faithful to the row.
    assert len(payload) == x.shape[1]
    for k, inp in enumerate(spec.description.input):
        assert payload[inp.name] == pytest.approx(float(row[k]))

    # FALSIFIABILITY: the exact payload the old code built must be REJECTED by
    # the same checker. Without this, the test above could pass vacuously.
    old_buggy_payload = {spec.description.input[0].name: np.asarray(x[0], dtype=np.float32)}
    problems = _payload_conformance_errors(old_buggy_payload, spec)
    assert problems, "the checker must reject the original one-array-under-one-key payload"
    assert any("declared scalar doubleType but got a sequence" in p for p in problems)
    assert any("!= declared inputs" in p for p in problems)


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

    # ONE SCALAR PER INPUT, via the SAME helper
    # `test_predict_payload_matches_the_declared_inputs` verifies on every
    # platform — so this marshalling is covered even where `predict` cannot
    # run. The previous version passed the whole 4-feature row under
    # `feature_0`, which the spec's four scalar doubleType inputs would have
    # rejected even on macOS.
    row = np.asarray(x[0], dtype=np.float64)
    out = loaded.predict(_predict_payload(loaded.get_spec(), row))
    prediction = np.asarray(next(iter(out.values())))
    assert np.all(np.isfinite(prediction))
