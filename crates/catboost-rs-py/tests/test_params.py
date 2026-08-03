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
_CORE_PY = (
    _REPO_ROOT
    / "catboost-master"
    / "catboost"
    / "python-package"
    / "catboost"
    / "core.py"
)


def _upstream_classifier_init_kwargs():
    """Extract every kwarg name from CatBoostClassifier.__init__ in core.py."""
    text = _CORE_PY.read_text()
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
    if not _CORE_PY.exists():
        pytest.skip(f"vendored core.py not found at {_CORE_PY}")
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
