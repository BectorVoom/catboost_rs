"""CM-PY Python-surface test: `save_coreml` on `CatBoostRegressor` (EXPORT-02).

Mirrors `test_onnx_export.py`'s unfitted-guard / fitted-write shape. There is
no Apple CoreML runtime on this Linux host (see
`.planning/plans/coreml-export/PLAN.md`'s verification caveat), so this is a
STRUCTURAL check only: the file is written and non-empty. An optional deeper
check runs if `coremltools` happens to be installed (`pytest.importorskip`,
same pattern `test_onnx_export.py` uses for `onnxruntime`).
"""

import pytest


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


def test_fitted_regressor_save_coreml_decodes_via_coremltools(tmp_path, toy_regression):
    """Optional deeper check: if `coremltools` is installed, the written file
    round-trips through its own protobuf loader and predicts finite values."""
    coremltools = pytest.importorskip(
        "coremltools",
        reason="coremltools is not installed in this environment; ad hoc "
        "install via `uv pip install --python .venv-py8/bin/python3 "
        "coremltools` to run the CoreML decode round-trip test locally",
    )
    import numpy as np
    import catboost_rs

    x, y = toy_regression
    model = catboost_rs.CatBoostRegressor(iterations=10, depth=3).fit(x, y)
    path = tmp_path / "regressor_rt.mlmodel"
    model.save_coreml(str(path))

    loaded = coremltools.models.MLModel(str(path))
    input_name = loaded.get_spec().description.input[0].name
    out = loaded.predict({input_name: np.asarray(x[0], dtype=np.float32)})
    prediction = next(iter(out.values()))
    assert np.all(np.isfinite(np.asarray(prediction)))
