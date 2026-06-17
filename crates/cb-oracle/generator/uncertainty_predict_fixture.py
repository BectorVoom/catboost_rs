#!/usr/bin/env python3
"""uncertainty_predict/ — the Plan 06.4-03 (LOSS-06, prediction-type half)
uncertainty PREDICTION oracle (Wave B).

This fixture freezes the three RMSEWithUncertainty uncertainty prediction types
the Rust `cb-model` apply path reproduces (06.4-RESEARCH Strand 2, D-6.4-04 — the
Phase-4 D-10 deferral closed):

  - RMSEWithUncertainty (single-model `predict`, NO virtual ensembles): 2 cols
        col0 = approx[0]                       (mean, identity)
        col1 = CalcSquaredExponent(approx[1])  = exp(2 * approx[1])  (variance)
    (`eval_helpers.cpp:422-427`; `eval_processing.h:47` — Pitfall 6: exp(2x),
    NOT exp(x) and NOT x^2.)

  - VirtEnsembles (`virtual_ensembles_predict`, virtual_ensembles_count=5):
    shape (n, V, 2). The V per-ensemble (mean, log-scale) pairs, with the
    log-scale dim transformed `exp(2 * x)` in place (`eval_helpers.cpp:428-444`).

  - TotalUncertainty (`virtual_ensembles_predict`, virtual_ensembles_count=5):
    3 cols `[mean, knowledgeUncertainty, dataUncertainty]` via
    `CalcRegressionUncertaitny` (`eval_helpers.cpp:209-269`, dimShift=2):
        mean[obj]      = (1/V) Σ_e approx[e*2][obj]
        knowledge[obj] = (1/V) Σ_e (approx[e*2][obj] - mean)^2     (epistemic)
        data[obj]      = (1/V) Σ_e exp(2 * approx[e*2+1][obj])     (aleatoric)

It REUSES the SAME RMSEWithUncertainty model SHAPE as Plan 06.4-02
(rmse_uncertainty_fixture.py): 11 iterations (>= 2*V+1 == 11 for
virtual_ensembles_count=5 — 06.4-RESEARCH Pitfall 5 / Open Q3: evalPeriod =
end // (2*V) must be > 0 and evalPeriod*V < end), model_shrink_rate=0 (A2 — the
default unshrinkCoef == 1 VE slice path), depth 3, Newton/1-iter, the D-07
isolating params, boost_from_average=False. The model is re-trained here (rather
than reloaded) so this generator is self-contained; it produces the SAME model
as the Plan-02 fixture (same params, same seed, same corpus).

OFFLINE / RUN-ONCE (D-6.4-02 / D-12): catboost==1.2.10 is NOT importable in CI;
this generator is run ONCE against `.venv` and the output is FROZEN-COMMITTED
under crates/cb-oracle/fixtures/uncertainty_predict/. The Rust oracle test loads
the frozen fixture; it NEVER calls catboost.

Run with:
    .venv/bin/python crates/cb-oracle/generator/uncertainty_predict_fixture.py
"""

import json
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor

GENERATOR_DIR = Path(__file__).resolve().parent
FIXTURES = GENERATOR_DIR.parent / "fixtures"
INPUTS = FIXTURES / "inputs"
UNCERTAINTY_PREDICT = FIXTURES / "uncertainty_predict"

CATBOOST_VERSION = "1.2.10"
SEED = 0

# >= 2*V + 1 trees for virtual_ensembles_count=5 (06.4-RESEARCH Pitfall 5 / Open
# Q3: evalPeriod = end // (2*V) must be > 0 and evalPeriod*V < end). 11 == 2*5+1
# is the minimum, matching the Plan-02 rmse_uncertainty fixture model shape.
ITERATIONS = 11
VIRTUAL_ENSEMBLES_COUNT = 5


def _assert_f64(arr: np.ndarray, name: str) -> np.ndarray:
    if arr.dtype != np.float64:
        raise AssertionError(f"{name} must be np.float64, got {arr.dtype}")
    return arr


def gen_uncertainty_predict() -> None:
    """Generate the uncertainty_predict/ prediction-type fixture (offline)."""
    UNCERTAINTY_PREDICT.mkdir(parents=True, exist_ok=True)
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

    # Persist the model.json (the SAME shape as the Plan-02 rmse_uncertainty
    # fixture): the apply path loads it, slices the trees, and reproduces the
    # uncertainty transforms. The per-dim bias `scale_and_bias[1]` = [mean,
    # 0.5*log(var)] is what the VE base ensemble (trees [0, begin)) carries.
    model.save_model(str(UNCERTAINTY_PREDICT / "model.json"), format="json")

    # --- Stage: RMSEWithUncertainty (single-model predict, 2 cols) ----------
    # col0 = mean (identity); col1 = variance = exp(2*log-scale) (Pitfall 6).
    rmse_unc = np.asarray(
        model.predict(x, prediction_type="RMSEWithUncertainty"), dtype=np.float64
    )
    if rmse_unc.shape != (x.shape[0], 2):
        raise AssertionError(
            f"RMSEWithUncertainty predict must be (n, 2), got {rmse_unc.shape}"
        )
    rmse_unc_flat = _assert_f64(rmse_unc.ravel().astype(np.float64), "rmse_with_uncertainty")
    np.save(
        UNCERTAINTY_PREDICT / "rmse_with_uncertainty.npy",
        rmse_unc_flat,
        allow_pickle=False,
    )

    # --- Stage: VirtEnsembles (V x 2 per object) ----------------------------
    # shape (n, V, 2): the per-ensemble (mean, variance=exp(2*log-scale)) pairs.
    virt = np.asarray(
        model.virtual_ensembles_predict(
            x,
            prediction_type="VirtEnsembles",
            virtual_ensembles_count=VIRTUAL_ENSEMBLES_COUNT,
        ),
        dtype=np.float64,
    )
    if virt.shape != (x.shape[0], VIRTUAL_ENSEMBLES_COUNT, 2):
        raise AssertionError(
            f"VirtEnsembles must be (n, {VIRTUAL_ENSEMBLES_COUNT}, 2), got {virt.shape}"
        )
    virt_flat = _assert_f64(virt.ravel().astype(np.float64), "virt_ensembles")
    np.save(UNCERTAINTY_PREDICT / "virt_ensembles.npy", virt_flat, allow_pickle=False)

    # --- Stage: TotalUncertainty (3 cols) -----------------------------------
    # [mean, knowledgeUncertainty (var of ensemble means), dataUncertainty
    #  (mean of exp(2*log-scale))] per CalcRegressionUncertaitny.
    total = np.asarray(
        model.virtual_ensembles_predict(
            x,
            prediction_type="TotalUncertainty",
            virtual_ensembles_count=VIRTUAL_ENSEMBLES_COUNT,
        ),
        dtype=np.float64,
    )
    if total.shape != (x.shape[0], 3):
        raise AssertionError(
            f"TotalUncertainty must be (n, 3), got {total.shape}"
        )
    total_flat = _assert_f64(total.ravel().astype(np.float64), "total_uncertainty")
    np.save(
        UNCERTAINTY_PREDICT / "total_uncertainty.npy", total_flat, allow_pickle=False
    )

    config = {
        "scenario": "uncertainty_predict",
        "requirement": "LOSS-06",
        "wave": 3,
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
        "n_iterations": ITERATIONS,
        "approx_dimension": 2,
        "virtual_ensembles_count": VIRTUAL_ENSEMBLES_COUNT,
        "prediction_types": ["RMSEWithUncertainty", "VirtEnsembles", "TotalUncertainty"],
        "rmse_with_uncertainty_layout": (
            "predict(prediction_type='RMSEWithUncertainty'): (n_rows, 2) "
            "OBJECT-MAJOR row-major [mean, variance], variance = exp(2*log-scale) "
            "(CalcSquaredExponent, Pitfall 6); flat f64."
        ),
        "virt_ensembles_layout": (
            "virtual_ensembles_predict(prediction_type='VirtEnsembles', "
            f"virtual_ensembles_count={VIRTUAL_ENSEMBLES_COUNT}): (n_rows, V, 2) "
            "row-major; per ensemble [mean, variance=exp(2*log-scale)]; flat f64."
        ),
        "total_uncertainty_layout": (
            "virtual_ensembles_predict(prediction_type='TotalUncertainty', "
            f"virtual_ensembles_count={VIRTUAL_ENSEMBLES_COUNT}): (n_rows, 3) "
            "OBJECT-MAJOR row-major [mean, knowledgeUncertainty, dataUncertainty]; "
            "flat f64. knowledge = (1/V)Σ(mean_e - mean)^2 (epistemic); "
            "data = (1/V)Σ exp(2*log-scale_e) (aleatoric)."
        ),
        "ve_slicing_note": (
            "ApplyVirtualEnsembles (apply.cpp:526-600): evalPeriod = end // (2*V) "
            "(integer); begin = end - evalPeriod*V; ensemble 0 seeds from trees "
            "[0, begin) (bias added here, treeStart==0); each ensemble adds the "
            "apply of trees [begin+v*evalPeriod, begin+(v+1)*evalPeriod) (no bias, "
            "treeStart>0) and copies the running sum forward (copyerLambda, "
            "copyToNextEnsemble). unshrinkCoef==1 (model_shrink_rate=0, A2)."
        ),
        "variance_transform_note": (
            "CalcSquaredExponent(x) = exp(2*x) (eval_processing.h:47, Pitfall 6) — "
            "NOT exp(x), NOT x^2. The log-scale dim is 0.5*log(variance)."
        ),
    }
    with (UNCERTAINTY_PREDICT / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
        fh.write("\n")


if __name__ == "__main__":
    gen_uncertainty_predict()
    print("Wrote uncertainty_predict oracle fixture (uncertainty_predict)")
