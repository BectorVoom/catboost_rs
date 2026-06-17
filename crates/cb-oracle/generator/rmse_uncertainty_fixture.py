#!/usr/bin/env python3
"""rmse_uncertainty/ — the Plan 06.4-02 (LOSS-08, loss half) RMSEWithUncertainty
per-stage training oracle (Wave B).

RMSEWithUncertainty is a 2-dimensional DIAGONAL-hessian loss riding the shipped
6.2 N-dim approx spine (D-6.4-04): dim 0 is the regression MEAN (approx[0]), dim 1
is the LOG-SCALE (approx[1]). It is a DECISIVELY CPU-Python-reachable, deterministic
loss (06.4-RESEARCH Strand 2 — verified live against `.venv` catboost==1.2.10):
`fit(loss_function='RMSEWithUncertainty')` trains on CPU and `predict(
prediction_type='RawFormulaVal')` returns the 2-dim raw approx, bit-reproducible
across runs with a pinned `random_seed` and `thread_count=1`.

der1/der2 (`error_functions.h:280-313`, `TRMSEWithUncertaintyError`,
`EHessianType::Diagonal`):
    diff    = target - approx[0]
    prec    = exp(-2 * approx[1])
    der1[0] = weight * diff
    der1[1] = weight * (diff*diff * prec - 1)
    der2[0] = -weight                       # diagonal
    der2[1] = -2 * weight * diff*diff * prec # diagonal

Trained as a CatBoostRegressor on the frozen `numeric_tiny` corpus (the RAW target
`y` — RMSEWithUncertainty's mean dim admits the full real line) with:
  - loss_function='RMSEWithUncertainty' (approx_dimension == 2),
  - leaf_estimation_method='Newton', leaf_estimation_iterations=1 (the upstream
    default for RMSEWithUncertainty, `catboost_options.cpp:77-82` — the diagonal
    hessian gives a PER-DIMENSION independent scalar Newton step, NOT the dense
    MultiClass symmetric solve; 06.4-RESEARCH Pitfall 4),
  - model_shrink_rate=0 (A2 — the default-path, unshrinkCoef == 1 for the Plan-03
    virtual-ensemble slice that reuses this model),
  - iterations=11 (>= 2*V+1 == 11 for the Plan-03 `virtual_ensembles_count=5`
    reuse — 06.4-RESEARCH Pitfall 5 / Open Q3),
  - the D-07 isolating params (boosting_type=Plain, bootstrap_type=No,
    random_strength=0, thread_count=1), depth 3, learning_rate 0.1, l2_leaf_reg 3.0,
    boost_from_average=False (zero-init both approx dims).

approx_dimension = 2. Predictions are RAW (RawFormulaVal, identity — NO link
transform; the variance transform exp(2*approx[1]) is a Plan-03 prediction-type
concern, not this loss oracle): staged.npy is staged_predict(RawFormulaVal) shape
(n, 2) object-major (A4); predictions.npy is predict(RawFormulaVal) shape (n, 2).
Gates Splits / LeafValues (LEAF-MAJOR) / StagedApprox (2-dim) / Predictions <= 1e-5
vs catboost 1.2.10 (thread_count=1).

OFFLINE / RUN-ONCE (D-6.4-02 / D-12): catboost==1.2.10 is NOT importable in CI;
this generator is run ONCE against `.venv` and the output is FROZEN-COMMITTED under
crates/cb-oracle/fixtures/rmse_uncertainty/. The Rust oracle test loads the frozen
fixture; it NEVER calls catboost.

Run with:
    .venv/bin/python crates/cb-oracle/generator/rmse_uncertainty_fixture.py
"""

import json
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor

GENERATOR_DIR = Path(__file__).resolve().parent
FIXTURES = GENERATOR_DIR.parent / "fixtures"
INPUTS = FIXTURES / "inputs"
RMSE_UNCERTAINTY = FIXTURES / "rmse_uncertainty"

CATBOOST_VERSION = "1.2.10"
SEED = 0

# >= 2*V + 1 trees so the Plan-03 virtual_ensembles_count=5 reuse has enough trees
# (06.4-RESEARCH Pitfall 5 / Open Q3: evalPeriod = end // (2*V) must be > 0 and
# evalPeriod*V < end). 11 == 2*5 + 1 is the minimum.
ITERATIONS = 11


def _assert_f64(arr: np.ndarray, name: str) -> np.ndarray:
    if arr.dtype != np.float64:
        raise AssertionError(f"{name} must be np.float64, got {arr.dtype}")
    return arr


def gen_rmse_uncertainty() -> None:
    """Generate the rmse_uncertainty/ training-stage fixture (offline, run-once)."""
    RMSE_UNCERTAINTY.mkdir(parents=True, exist_ok=True)
    x = np.load(INPUTS / "numeric_tiny" / "X.npy")
    y = np.load(INPUTS / "numeric_tiny" / "y.npy")

    params = {
        "iterations": ITERATIONS,
        "learning_rate": 0.1,
        "depth": 3,
        "l2_leaf_reg": 3.0,
        "bootstrap_type": "No",
        "boosting_type": "Plain",
        "random_strength": 0,
        "leaf_estimation_iterations": 1,
        "leaf_estimation_method": "Newton",  # RMSEWithUncertainty default (Pitfall 4).
        "model_shrink_rate": 0,  # A2: default path, unshrinkCoef == 1.
        "score_function": "L2",  # CPU-supported; simplest split math.
        "random_seed": SEED,
        "thread_count": 1,  # Deterministic summation order.
        "verbose": False,
        "loss_function": "RMSEWithUncertainty",
        "boost_from_average": False,
    }
    model = CatBoostRegressor(**params)
    model.fit(x, y)

    # --- Stage: Splits + LeafValues (model.json) ----------------------------
    model.save_model(str(RMSE_UNCERTAINTY / "model.json"), format="json")

    # --- Stage: StagedApprox (RawFormulaVal, shape (n, 2) object-major) ------
    staged = [
        np.asarray(p, dtype=np.float64)
        for p in model.staged_predict(x, prediction_type="RawFormulaVal")
    ]
    staged_flat = _assert_f64(
        np.concatenate([s.ravel() for s in staged]).astype(np.float64), "staged"
    )
    np.save(RMSE_UNCERTAINTY / "staged.npy", staged_flat, allow_pickle=False)

    # --- Stage: Predictions (RAW 2-dim, identity, (n, 2) object-major) -------
    preds = np.asarray(
        model.predict(x, prediction_type="RawFormulaVal"), dtype=np.float64
    )
    preds_flat = _assert_f64(preds.ravel().astype(np.float64), "predictions")
    np.save(RMSE_UNCERTAINTY / "predictions.npy", preds_flat, allow_pickle=False)

    n_iter = len(staged)
    dim = int(np.asarray(staged[0]).shape[1]) if n_iter else 0
    if dim != 2:
        raise AssertionError(
            f"RMSEWithUncertainty must produce approx_dimension == 2, got {dim}"
        )
    if n_iter != ITERATIONS:
        raise AssertionError(
            f"expected {ITERATIONS} staged iterations, got {n_iter}"
        )

    config = {
        "scenario": "rmse_uncertainty",
        "requirement": "LOSS-08",
        "wave": 2,
        "seed": SEED,
        "catboost_version": CATBOOST_VERSION,
        "thread_count": 1,
        "input_dataset": "numeric_tiny",
        "loss_function": "RMSEWithUncertainty",
        "leaf_estimation_method": "Newton",
        "leaf_estimation_iterations": 1,
        "model_shrink_rate": 0,
        "score_function": "L2",
        "boost_from_average": False,
        "params": params,
        "n_rows": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "n_iterations": n_iter,
        "approx_dimension": dim,
        "stages": ["Splits", "LeafValues", "StagedApprox", "Predictions"],
        "staged_layout": (
            "staged_predict(RawFormulaVal): per-iter (n_rows, 2) OBJECT-MAJOR "
            "(row-major object then dim: [mean, log-scale]), concatenated across "
            "iterations; flat f64."
        ),
        "predictions_layout": (
            "predict(RawFormulaVal): (n_rows, 2) OBJECT-MAJOR row-major (RAW 2-dim "
            "[mean, log-scale] approx, identity — NO variance transform), flat f64."
        ),
        "leaf_values_layout": (
            "model.json leaf_values are LEAF-MAJOR (leaf0_d0, leaf0_d1, leaf1_d0, "
            "...) length leaves*2; leaf_weights length leaves."
        ),
        "prediction_type": "RawFormulaVal",
        "leaf_method_note": (
            "RMSEWithUncertainty=Newton/1-iter (DIAGONAL hessian: each dim is an "
            "INDEPENDENT scalar Newton step der2[0]=-w, der2[1]=-2*w*diff^2*prec; "
            "NOT the MultiClass dense symmetric solve, 06.4-RESEARCH Pitfall 4)."
        ),
        "der_note": (
            "RMSEWithUncertainty der (error_functions.h:280-313): "
            "diff=target-approx[0]; prec=exp(-2*approx[1]); "
            "der1=[w*diff, w*(diff*diff*prec-1)]; "
            "der2-diag=[-w, -2*w*diff*diff*prec]."
        ),
    }
    with (RMSE_UNCERTAINTY / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
        fh.write("\n")


if __name__ == "__main__":
    gen_rmse_uncertainty()
    print("Wrote rmse_uncertainty oracle fixture (rmse_uncertainty)")
