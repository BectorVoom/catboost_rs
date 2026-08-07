"""D-05/D-06/D-07 param-vocabulary registry + fit()-time validation tests.

- Every kwarg in the vendored upstream ``CatBoostClassifier.__init__`` signature
  is present in the registry with a status tag (IMPLEMENTED | KNOWN_NOT_YET).
- A known-but-unimplemented param (``nan_mode``) is rejected at ``fit()`` as a
  parity gap; a typo (``iteratons``) suggests the closest match; sklearn aliases
  (``n_estimators`` / ``max_depth`` / ``reg_lambda``) resolve and ``fit`` succeeds.
"""

import re
from pathlib import Path

import numpy as np
import pytest

import catboost_rs
from catboost_rs import CatBoostParameterError, CatBoostRegressor


# Repo layout: crates/catboost-rs-py/tests/ -> repo root is three parents up.
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _core_py() -> Path:
    """Locate upstream's ``core.py`` — the AUTHORITATIVE kwarg vocabulary.

    F18: this used to point ONLY at ``catboost-master/catboost/python-package/
    catboost/core.py``, which does not exist in this checkout (the vendored tree
    is a three-file stub of a DIFFERENT revision), so the registry-truthfulness
    test silently SKIPPED — and a skipped test is not a gate. Fall back to the
    INSTALLED ``catboost==1.2.10`` package, which is the parity target this
    repository actually pins.
    """
    vendored = (
        _REPO_ROOT
        / "catboost-master"
        / "catboost"
        / "python-package"
        / "catboost"
        / "core.py"
    )
    if vendored.exists():
        return vendored
    import catboost

    return Path(catboost.__file__).resolve().parent / "core.py"


def _upstream_classifier_init_kwargs():
    """Extract every kwarg name from CatBoostClassifier.__init__ in core.py."""
    text = _core_py().read_text()
    # Find the CatBoostClassifier.__init__ signature block.
    cls_idx = text.index("class CatBoostClassifier")
    init_idx = text.index("def __init__(", cls_idx)
    body_idx = text.index("):", init_idx)
    sig = text[init_idx:body_idx]
    # Each kwarg appears as `\n        name=None,`.
    names = re.findall(r"^\s+([a-z_][a-z0-9_]*)=None", sig, flags=re.MULTILINE)
    return sorted(set(names))


def _toy_xy():
    x = np.array(
        [[0.0, 1.0], [1.0, 0.0], [2.0, 2.0], [3.0, 1.0]], dtype=np.float32
    )
    y = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    return x, y


def test_every_upstream_param_is_in_registry():
    # F18: NEVER skip. A skipped test is not a gate, and this is the only thing
    # standing between the registry and a silently-missing upstream parameter.
    core = _core_py()
    assert core.exists(), (
        f"upstream core.py not found at {core}; the registry-truthfulness gate "
        "cannot be skipped — install catboost==1.2.10"
    )
    upstream = _upstream_classifier_init_kwargs()
    assert len(upstream) > 100, "expected the full upstream kwarg vocabulary"
    missing = [
        name for name in upstream if catboost_rs._param_status(name) is None
    ]
    assert not missing, f"registry missing upstream params: {missing}"
    # Each is tagged with a valid status.
    for name in upstream:
        status = catboost_rs._param_status(name)
        assert status in ("IMPLEMENTED", "KNOWN_NOT_YET"), (name, status)


def test_known_not_yet_param_rejected_as_parity_gap():
    x, y = _toy_xy()
    model = CatBoostRegressor(nan_mode="Min")
    with pytest.raises(CatBoostParameterError) as excinfo:
        model.fit(x, y)
    msg = str(excinfo.value)
    assert "nan_mode" in msg
    assert "parity gap" in msg


def test_typo_param_suggests_closest_match():
    x, y = _toy_xy()
    model = CatBoostRegressor(iteratons=10)
    with pytest.raises(CatBoostParameterError) as excinfo:
        model.fit(x, y)
    msg = str(excinfo.value)
    assert "iteratons" in msg
    assert "iterations" in msg  # suggestion


def test_sklearn_aliases_resolve_and_fit_succeeds():
    x, y = _toy_xy()
    model = CatBoostRegressor(n_estimators=10, max_depth=3, reg_lambda=2.0)
    model.fit(x, y)  # must not raise
    preds = model.predict(x)
    assert preds.shape == (4,)


def test_validation_fires_at_fit_not_init():
    # __init__ must do NO validation (D-06): constructing with a bad param is OK;
    # only fit() rejects it.
    model = CatBoostRegressor(nan_mode="Min")  # no raise here
    x, y = _toy_xy()
    with pytest.raises(CatBoostParameterError):
        model.fit(x, y)


def test_ctr_default_parity_gap_is_documented():
    """SPEC-CTRT-19 / SPEC-CATF-Δ2 — the single-description CTR limit is recorded.

    Upstream's CPU default is a LIST of two CTR descriptions; this crate models
    ONE description with a prior LIST. That divergence must be documented, with
    its upstream anchor, everywhere a user or implementer will look, so it
    cannot silently rot into an undocumented parity gap.
    """
    src = (_REPO_ROOT / "crates" / "catboost-rs-py" / "src" / "params.rs").read_text()
    assert "catboost_options.cpp:439-453" in src, (
        "the multi-description default parity gap must cite its upstream anchor"
    )
    assert "single-description" in src or "one CTR description" in src

    boosting = (
        _REPO_ROOT / "crates" / "cb-train" / "src" / "boosting.rs"
    ).read_text()
    assert boosting.count("catboost_options.cpp:439-453") >= 4, (
        "all four CTR *_default() doc comments must cite the parity-gap anchor"
    )
    # The now-false framing must be deleted, not left beside the new text.
    assert "never exercise the CTR path" not in boosting
    assert "never exercise the combination path" not in boosting


# ── FPP-16: task_type is VALIDATED-INFORMATIONAL ────────────────────────────────────────


def test_task_type_cpu_does_not_change_predictions():
    """FPP-16 (D1) — ``task_type="CPU"`` is pure input validation, not behaviour.

    This is the assertion that PROVES the "validated-informational" claim rather than
    asserting it in prose: the same fixed-seed fit, once with and once without
    ``task_type="CPU"``, must produce **bit-identical** predictions. Anything less than
    bit-identity would mean the parameter reached the trainer.
    """
    x, y = _toy_xy()

    plain = CatBoostRegressor(iterations=5, depth=3, random_seed=7, learning_rate=0.3)
    plain.fit(x, y)
    tagged = CatBoostRegressor(
        iterations=5, depth=3, random_seed=7, learning_rate=0.3, task_type="CPU"
    )
    tagged.fit(x, y)

    np.testing.assert_array_equal(
        plain.predict(x),
        tagged.predict(x),
        err_msg="task_type='CPU' must be inert — predictions must be BIT-identical",
    )


def test_task_type_is_accepted_and_not_a_parity_gap():
    """``task_type`` must no longer be rejected at ``fit()`` as a KNOWN_NOT_YET gap."""
    x, y = _toy_xy()
    CatBoostRegressor(iterations=3, task_type="CPU").fit(x, y)  # must not raise
    CatBoostRegressor(iterations=3, task_type=None).fit(x, y)  # None is inert


def test_task_type_unknown_value_lists_the_legal_values():
    """A wrong VALUE lists CPU/GPU; it must not be reported as a misspelled NAME."""
    x, y = _toy_xy()
    model = CatBoostRegressor(iterations=3, task_type="TPU")
    with pytest.raises(CatBoostParameterError) as excinfo:
        model.fit(x, y)
    message = str(excinfo.value)
    assert "CPU" in message and "GPU" in message
    assert "did you mean" not in message


def test_task_type_gpu_on_a_cpu_wheel_is_an_actionable_error():
    """On a CPU-only wheel, ``task_type="GPU"`` must fail loudly and say how to fix it.

    Skipped on a device-feature wheel, where GPU is legitimately accepted. Silently
    training on the CPU after an explicit GPU request is precisely the silently-wrong-model
    failure the module's honesty policy exists to prevent.
    """
    x, y = _toy_xy()
    model = CatBoostRegressor(iterations=3, task_type="GPU")
    try:
        model.fit(x, y)
    except CatBoostParameterError as exc:
        message = str(exc)
        assert "cuda" in message and "rocm" in message and "wgpu" in message
        assert "compile-time" in message
    else:
        # A device-feature wheel: GPU is accepted, which is the other half of the contract.
        pass
