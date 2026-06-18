#!/usr/bin/env python3
"""Generate the FEAT-03 monotone-constraint oracle fixture (Phase 06.6 plan 02).

RUN-ONCE / COMMIT discipline (mirrors gen_penalty_fixtures.py): a BUILD-TIME tool
run OUTSIDE CI that writes committed frozen fixtures. CI only READS them; this
generator is NEVER invoked from CI. Re-run by hand only to regenerate, then
COMMIT the result.

A SymmetricTree (oblivious) RMSE regressor trained on the frozen `numeric_tiny`
INPUT corpus (50 rows, 4 float features) with the first-slice simplified
isolating params (bootstrap_type='No', random_strength=0, score_function='L2',
leaf_estimation_method='Gradient', thread_count=1, fixed random_seed) so a
divergence is attributable ONLY to the monotone-constraint leaf-value
post-pass (the isotonic PAVA projection):

    monotone/increasing_decreasing -> monotone_constraints=[-1, 0, 1, 0]
        feature 0 NON-INCREASING (-1), feature 2 NON-DECREASING (+1),
        features 1 & 3 free (0).

CRITICAL — `model_shrink_rate=0` is pinned EXPLICITLY. CatBoost AUTO-enables
model shrinkage (a per-tree multiplicative decay of all prior leaf values,
`model_shrink_rate = learning_rate * <factor>`, `model_shrink_mode=Constant`)
the moment monotone constraints are present, which would confound the isotonic
projection with an unrelated shrinkage drift (~1e-3 per tree). Pinning it to 0
ISOLATES the FEAT-03 leaf-value post-pass as the only difference vs an
unconstrained model — the oracle then tests the PAVA math and nothing else.

The constraints are chosen so the projection GENUINELY BINDS (the unconstrained
leaf values violate them and are pooled, max leaf diff ~3.9e-1 vs the
unconstrained model), so the fixture is non-vacuous. Because the constrained
approx feeds back into later trees' gradients, the chosen SPLITS of the
constrained model may differ from the UNCONSTRAINED model — that is expected;
the oracle locks our trainer against THIS (monotone) fixture's own splits +
leaf values, which is self-consistent.

Monotone constraints are a PUBLIC catboost 1.2.10 training param (D-6.6-08) and
are OBLIVIOUS-ONLY (SymmetricTree) — upstream rejects them under every
non-symmetric grow policy (`monotonic_constraint_utils.h:42`). They are enforced
as an isotonic (PAVA) projection over the per-leaf DELTAS during leaf estimation
(`CalcMonotonicLeafDeltasSimple`, `approx_calcer.cpp:551`), AFTER the structure is
built — so the SPLITS are UNAFFECTED (assert them too as a sanity lock) and only
the LEAF VALUES change versus an unconstrained model.

For the scenario:
    save_model(format=json)  -> model.json    (splits + leaf_values)
    staged_predict()         -> staged.npy    (n_iterations x n_rows, flat f64)
    predict()                -> predictions.npy

Layout matches the existing cb-oracle fixtures so the cb-train monotone oracle
test reuses the `fixture()` / `load_model_json` / `compare_stage` harness.

Run (after gen_inputs.py / gen_fixtures.py have produced numeric_tiny):
    .venv/bin/python crates/cb-oracle/generator/gen_monotone_fixtures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor

GENERATOR_DIR = Path(__file__).resolve().parent
FIXTURES = GENERATOR_DIR.parent / "fixtures"
INPUTS = FIXTURES / "inputs"
MONOTONE = FIXTURES / "monotone"

CATBOOST_VERSION = "1.2.10"
SEED = 0

# First-slice simplified isolating params (mirrors gen_penalty_fixtures).
# SymmetricTree (the default grow_policy) — FEAT-03 monotone is oblivious-only.
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
    # Pin OFF the model shrinkage CatBoost auto-enables under monotone constraints
    # so the isotonic (PAVA) leaf-value projection is the ONLY difference vs an
    # unconstrained model (see module docstring).
    "model_shrink_rate": 0,
    "verbose": False,
}


def _assert_f64(arr: np.ndarray, name: str) -> np.ndarray:
    if arr.dtype != np.float64:
        raise TypeError(f"{name} must be float64, got {arr.dtype}")
    return arr


def gen_monotone_fixtures() -> None:
    MONOTONE.mkdir(parents=True, exist_ok=True)

    x = np.load(INPUTS / "numeric_tiny" / "X.npy")
    y = np.load(INPUTS / "numeric_tiny" / "y.npy")
    n_features = int(x.shape[1])

    name = "increasing_decreasing"
    monotone = [-1, 0, 1, 0]

    scenario_dir = MONOTONE / name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    params = {**ISOLATING_PARAMS, "monotone_constraints": monotone}
    model = CatBoostRegressor(**params)
    model.fit(x, y)

    # Sanity (NOT committed): the constrained leaf values must DIFFER from the
    # unconstrained model — otherwise the fixture would pass even if the PAVA
    # post-pass were a no-op (a false-confidence oracle).
    baseline = CatBoostRegressor(**ISOLATING_PARAMS)
    baseline.fit(x, y)
    if np.allclose(model.predict(x), baseline.predict(x), atol=1e-9):
        raise SystemExit(
            "FIXTURE VACUOUS: monotone predictions equal unconstrained — "
            "choose constraints that actually bind"
        )

    # Stage: Splits + LeafValues (model.json).
    model.save_model(str(scenario_dir / "model.json"), format="json")

    # Stage: StagedApprox (per-iteration raw approximant).
    staged = [np.asarray(p, dtype=np.float64) for p in model.staged_predict(x)]
    staged_flat = _assert_f64(
        np.concatenate([s.ravel() for s in staged]).astype(np.float64), "staged"
    )
    np.save(scenario_dir / "staged.npy", staged_flat, allow_pickle=False)

    # Stage: Predictions (final raw approximant).
    preds = _assert_f64(np.asarray(model.predict(x), dtype=np.float64), "predictions")
    np.save(scenario_dir / "predictions.npy", preds, allow_pickle=False)

    config = {
        "scenario": f"monotone/{name}",
        "seed": SEED,
        "catboost_version": CATBOOST_VERSION,
        "thread_count": 1,
        "input_dataset": "numeric_tiny",
        "loss_function": "RMSE",
        "grow_policy": "SymmetricTree",
        "monotone_constraints": monotone,
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
        "monotone_note": (
            "FEAT-03. monotone_constraints [+1/-1/0 per feature] is enforced as an "
            "isotonic (PAVA) projection over the per-leaf DELTAS during leaf "
            "estimation (CalcMonotonicLeafDeltasSimple, approx_calcer.cpp:551), AFTER "
            "the structure is built. The SPLITS are UNAFFECTED (a sanity lock); only "
            "the LEAF VALUES change vs an unconstrained model. Oblivious-only "
            "(SymmetricTree); upstream rejects monotone under non-symmetric policies."
        ),
    }
    with (scenario_dir / "config.json").open("w") as fh:
        json.dump(config, fh, indent=2)

    print(f"monotone/{name}: {len(staged)} iters, n_rows={x.shape[0]}, constraints={monotone}")


if __name__ == "__main__":
    gen_monotone_fixtures()
    print("monotone fixtures generated under", MONOTONE)
