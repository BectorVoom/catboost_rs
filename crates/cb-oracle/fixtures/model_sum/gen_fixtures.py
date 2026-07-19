#!/usr/bin/env python3
"""Offline fixture generator for `sum_models` (SPEC `sum_models`, SM-07).

Generates, from catboost==1.2.10 in the project `.venv`:
  - two FLOAT-ONLY (no categorical features -> ctr_data-free, deterministic
    to apply) oblivious `CatBoostRegressor`s, `m0` and `m1`;
  - the fixed `numeric_tiny` eval matrix `X.npy`;
  - each model's own upstream `RawFormulaVal` prediction (`expected_m0.npy`,
    `expected_m1.npy`) -- isolates apply from merge arithmetic in the Rust
    oracle test;
  - upstream `catboost.sum_models([m0, m1], weights).predict(X,
    prediction_type="RawFormulaVal")` for two weight vectors
    (`expected_w_1_1.npy`, `expected_w_03_07.npy`).

`m0`/`m1` are trained with `depth=1, iterations=1` (a single tree, one split
each) and DIFFERENT `learning_rate` (0.1 vs 0.6) -- everything else (seed,
`l2_leaf_reg`, `score_function`, `boost_from_average`) held identical. This is
the SPEC `sum_models` first-slice precondition (SM-04): the merge REQUIRES
`float_feature_borders` to match byte-for-byte across inputs. Empirically
(verified against installed catboost==1.2.10), a single greedy split's
CANDIDATE-BEST choice at iteration 0 is independent of `learning_rate` (it
only rescales the resulting leaf value / Newton step), so the two models are
GENUINELY DIFFERENT (distinct leaf values, distinct `.cbm` bytes) while
sharing IDENTICAL borders and bias -- the honest way to satisfy SM-04's
precondition with two real, independently-trained models rather than saving
one model twice. Deeper/`iterations>1` configurations were tried and reliably
diverge in which borders get baked into the model (CatBoost only serializes
borders actually used by a split), which would violate SM-04 and is exactly
why this first slice defers cross-model-border merging (SPEC Sec. 2).

Run (from repo root):
    .venv/bin/python crates/cb-oracle/fixtures/model_sum/gen_fixtures.py

Pinned seeds, thread_count=1, bootstrap_type="No". No fabrication -- every
vector comes straight from `catboost.sum_models(...).predict(...)`.

`prediction_type="RawFormulaVal"` is passed EXPLICITLY on every `.predict(...)`
call below (per-model AND summed) so the fixture is unambiguous even though
`CatBoostRegressor.predict` already returns raw values by default (SPEC SM-07
prediction-type pin).
"""
import json
import os

import numpy as np
import catboost as cb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
INPUTS = os.path.join(ROOT, "crates", "cb-oracle", "fixtures", "inputs", "numeric_tiny")

WEIGHTS_1_1 = [1.0, 1.0]
WEIGHTS_03_07 = [0.3, 0.7]


def _numeric_regression_pool():
    """The `numeric_tiny` X with its RAW continuous y (regression target).
    Numeric features ONLY (no cat features) -> both models are CTR-free."""
    X = np.load(os.path.join(INPUTS, "X.npy"))
    y = np.load(os.path.join(INPUTS, "y.npy")).astype(np.float64)
    return X, y, cb.Pool(X, y)


def _train(learning_rate, pool):
    # depth=1, iterations=1: the single greedy split at iteration 0 is chosen
    # independently of `learning_rate` (see module docstring) -- the SAME
    # border/bias is baked into both models, only the leaf values (and hence
    # the `.cbm` bytes) differ, satisfying SM-04's identical-borders
    # precondition with two genuinely differently-trained models.
    m = cb.CatBoostRegressor(
        loss_function="RMSE",
        boost_from_average=True,
        bootstrap_type="No",
        depth=1,
        iterations=1,
        l2_leaf_reg=3.0,
        leaf_estimation_iterations=1,
        leaf_estimation_method="Gradient",
        learning_rate=learning_rate,
        random_seed=0,
        random_strength=0,
        score_function="L2",
        thread_count=1,
        verbose=False,
    )
    m.fit(pool)
    return m


def main():
    X, y, pool = _numeric_regression_pool()

    # Two float-only oblivious regressors sharing IDENTICAL borders/bias
    # (SM-04 precondition) but DIFFERENT leaf values (different
    # `learning_rate`) -- see module docstring for why depth=1/iterations=1.
    m0 = _train(learning_rate=0.1, pool=pool)
    m1 = _train(learning_rate=0.6, pool=pool)

    m0.save_model(os.path.join(HERE, "m0.cbm"), format="cbm")
    m1.save_model(os.path.join(HERE, "m1.cbm"), format="cbm")

    np.save(os.path.join(HERE, "X.npy"), X.astype(np.float64))

    expected_m0 = np.asarray(
        m0.predict(X, prediction_type="RawFormulaVal"), dtype=np.float64
    )
    expected_m1 = np.asarray(
        m1.predict(X, prediction_type="RawFormulaVal"), dtype=np.float64
    )
    np.save(os.path.join(HERE, "expected_m0.npy"), expected_m0)
    np.save(os.path.join(HERE, "expected_m1.npy"), expected_m1)

    # R2 (SPEC Sec.9): confirmed against the installed catboost==1.2.10 —
    # `sum_models(models, weights=None, ctr_merge_policy='IntersectingCountersAverage')`.
    # Both models are float-only, so `ctr_merge_policy` is irrelevant here.
    merged_1_1 = cb.sum_models([m0, m1], weights=WEIGHTS_1_1)
    merged_03_07 = cb.sum_models([m0, m1], weights=WEIGHTS_03_07)

    expected_w_1_1 = np.asarray(
        merged_1_1.predict(X, prediction_type="RawFormulaVal"), dtype=np.float64
    )
    expected_w_03_07 = np.asarray(
        merged_03_07.predict(X, prediction_type="RawFormulaVal"), dtype=np.float64
    )
    np.save(os.path.join(HERE, "expected_w_1_1.npy"), expected_w_1_1)
    np.save(os.path.join(HERE, "expected_w_03_07.npy"), expected_w_03_07)

    config = {
        "catboost_version": cb.__version__,
        "input_dataset": "numeric_tiny",
        "scenario": "model_sum",
        "thread_count": 1,
        "prediction_type": "RawFormulaVal",
        "upstream_signature": "sum_models(models, weights=None, ctr_merge_policy='IntersectingCountersAverage')",
        "m0_params": {
            "loss_function": "RMSE",
            "boost_from_average": True,
            "bootstrap_type": "No",
            "depth": 1,
            "iterations": 1,
            "l2_leaf_reg": 3.0,
            "leaf_estimation_iterations": 1,
            "leaf_estimation_method": "Gradient",
            "learning_rate": 0.1,
            "random_seed": 0,
            "random_strength": 0,
            "score_function": "L2",
            "thread_count": 1,
        },
        "m1_params": {
            "loss_function": "RMSE",
            "boost_from_average": True,
            "bootstrap_type": "No",
            "depth": 1,
            "iterations": 1,
            "l2_leaf_reg": 3.0,
            "leaf_estimation_iterations": 1,
            "leaf_estimation_method": "Gradient",
            "learning_rate": 0.6,
            "random_seed": 0,
            "random_strength": 0,
            "score_function": "L2",
            "thread_count": 1,
        },
        "weights_1_1": WEIGHTS_1_1,
        "weights_03_07": WEIGHTS_03_07,
        "note": (
            "sum_models (SPEC sum_models, SM-07): two float-only (no cat "
            "features -> ctr_data-free) oblivious RMSE regressors sharing "
            "IDENTICAL float_feature_borders and bias (SM-04's first-slice "
            "merge precondition) but DIFFERENT leaf values (different "
            "learning_rate at depth=1/iterations=1 -- see gen_fixtures.py "
            "module docstring for why deeper/more-iterations configurations "
            "reliably diverge in which borders get baked into the model). "
            "expected_m0/expected_m1 isolate apply from merge arithmetic; "
            "expected_w_1_1/expected_w_03_07 are catboost.sum_models(...).predict(...) "
            "ground truth. Every .predict(...) call pins "
            "prediction_type='RawFormulaVal' explicitly."
        ),
        "artifacts": [
            "m0.cbm",
            "m1.cbm",
            "X.npy",
            "expected_m0.npy",
            "expected_m1.npy",
            "expected_w_1_1.npy",
            "expected_w_03_07.npy",
        ],
    }
    with open(os.path.join(HERE, "config.json"), "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    print("expected_m0:", expected_m0.tolist())
    print("expected_m1:", expected_m1.tolist())
    print("expected_w_1_1:", expected_w_1_1.tolist())
    print("expected_w_03_07:", expected_w_03_07.tolist())


if __name__ == "__main__":
    main()
