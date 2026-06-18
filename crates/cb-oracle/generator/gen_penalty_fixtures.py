#!/usr/bin/env python3
"""Generate the FEAT-04 feature-penalty oracle fixtures (Phase 06.6 plan 01).

RUN-ONCE / COMMIT discipline (mirrors gen_fixtures.py): a BUILD-TIME tool run
OUTSIDE CI that writes committed frozen fixtures. CI only READS them; this
generator is NEVER invoked from CI. Re-run by hand only to regenerate, then
COMMIT the result.

Three penalty kinds, each a SymmetricTree (oblivious) RMSE regressor trained on
the frozen `numeric_tiny` INPUT corpus (50 rows, 4 float features) with the
first-slice simplified isolating params (bootstrap_type='No', random_strength=0,
score_function='L2', leaf_estimation_method='Gradient', thread_count=1, fixed
random_seed) so a divergence is attributable ONLY to the penalty math:

    penalty/feature_weights   -> feature_weights=[1, 0.1, 1, 1]
    penalty/first_use         -> first_feature_use_penalties=[0, 5, 0, 0],
                                 penalties_coefficient=1
    penalty/per_object        -> per_object_feature_penalties=[0, 0.1, 0, 0],
                                 penalties_coefficient=1

All three are PUBLIC catboost 1.2.10 training params (D-6.6-08), so the Python
API output IS the ground truth. Each penalty vector is chosen to actively change
the chosen splits relative to the unpenalized model (feature 1 is the natural
favourite without penalties), so the fixture genuinely exercises the penalty
path rather than trivially matching the default.

For each scenario:
    save_model(format=json)  -> model.json    (splits + leaf_values)
    staged_predict()         -> staged.npy    (n_iterations x n_rows, flat f64)
    predict()                -> predictions.npy

Layout matches the existing cb-oracle fixtures so the cb-train penalty oracle
test reuses the `fixture()` / `load_model_json` / `compare_stage` harness.

Run (after gen_inputs.py / gen_fixtures.py have produced numeric_tiny):
    .venv/bin/python crates/cb-oracle/generator/gen_penalty_fixtures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor

GENERATOR_DIR = Path(__file__).resolve().parent
FIXTURES = GENERATOR_DIR.parent / "fixtures"
INPUTS = FIXTURES / "inputs"
PENALTY = FIXTURES / "penalty"

CATBOOST_VERSION = "1.2.10"
SEED = 0

# First-slice simplified isolating params (mirrors gen_fixtures.ISOLATING_PARAMS).
# SymmetricTree (the default grow_policy) — FEAT-04 rides the oblivious grower.
ISOLATING_PARAMS = {
    "iterations": 5,
    "learning_rate": 0.1,
    "depth": 2,
    "l2_leaf_reg": 3.0,
    "bootstrap_type": "No",
    "random_strength": 0,
    "leaf_estimation_iterations": 1,
    "score_function": "L2",
    "leaf_estimation_method": "Gradient",
    "grow_policy": "SymmetricTree",
    "random_seed": SEED,
    "thread_count": 1,
    "boost_from_average": True,
    "verbose": False,
}


def _assert_f64(arr: np.ndarray, name: str) -> np.ndarray:
    if arr.dtype != np.float64:
        raise TypeError(f"{name} must be float64, got {arr.dtype}")
    return arr


def gen_penalty_fixtures() -> None:
    PENALTY.mkdir(parents=True, exist_ok=True)

    x = np.load(INPUTS / "numeric_tiny" / "X.npy")
    y = np.load(INPUTS / "numeric_tiny" / "y.npy")
    n_features = int(x.shape[1])

    # (scenario_name, extra penalty params).
    scenarios = [
        (
            "feature_weights",
            {"feature_weights": [1.0, 0.1, 1.0, 1.0]},
        ),
        (
            "first_use",
            {
                "first_feature_use_penalties": [0.0, 5.0, 0.0, 0.0],
                "penalties_coefficient": 1.0,
            },
        ),
        (
            "per_object",
            {
                "per_object_feature_penalties": [0.0, 0.1, 0.0, 0.0],
                "penalties_coefficient": 1.0,
            },
        ),
    ]

    for name, penalty_params in scenarios:
        scenario_dir = PENALTY / name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        params = {**ISOLATING_PARAMS, **penalty_params}
        model = CatBoostRegressor(**params)
        model.fit(x, y)

        # Stage: Splits + LeafValues (model.json).
        model.save_model(str(scenario_dir / "model.json"), format="json")

        # Stage: StagedApprox (per-iteration raw approximant).
        staged = [np.asarray(p, dtype=np.float64) for p in model.staged_predict(x)]
        staged_flat = _assert_f64(
            np.concatenate([s.ravel() for s in staged]).astype(np.float64), "staged"
        )
        np.save(scenario_dir / "staged.npy", staged_flat, allow_pickle=False)

        # Stage: Predictions (final raw approximant).
        preds = _assert_f64(
            np.asarray(model.predict(x), dtype=np.float64), "predictions"
        )
        np.save(scenario_dir / "predictions.npy", preds, allow_pickle=False)

        config = {
            "scenario": f"penalty/{name}",
            "seed": SEED,
            "catboost_version": CATBOOST_VERSION,
            "thread_count": 1,
            "input_dataset": "numeric_tiny",
            "loss_function": "RMSE",
            "grow_policy": "SymmetricTree",
            "params": params,
            "n_rows": int(x.shape[0]),
            "n_features": n_features,
            "n_iterations": len(staged),
            "prediction_type": "RawFormulaVal",
            "stages": ["Splits", "LeafValues", "StagedApprox", "Predictions"],
            "staged_layout": (
                "flat f64: stage 0 (n_rows), then stage 1, ... ; n_iterations "
                "stages (raw approximant)"
            ),
            "penalty_note": (
                "FEAT-04. feature_weights is a MULTIPLICATIVE candidate-gain "
                "factor (GetSplitFeatureWeight); first_feature_use_penalties and "
                "per_object_feature_penalties are SUBTRACTIVE (PenalizeBestSplits), "
                "scaled by penalties_coefficient, applied while the feature is "
                "unused. Vectors target feature index 1 (the unpenalized favourite) "
                "so the penalty actively changes the chosen splits."
            ),
        }
        with (scenario_dir / "config.json").open("w") as fh:
            json.dump(config, fh, indent=2)

        print(f"penalty/{name}: {len(staged)} iters, n_rows={x.shape[0]}")


if __name__ == "__main__":
    gen_penalty_fixtures()
    print("penalty fixtures generated under", PENALTY)
