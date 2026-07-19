#!/usr/bin/env python3
"""Offline fixture generator for `staged_predict` (SP-04, R1).

Freezes, from catboost==1.2.10 in the project `.venv`:
  - a float-only oblivious `CatBoostRegressor` `.cbm` (pinned seed,
    thread_count=1, bootstrap_type="No") trained on `inputs/numeric_tiny/X.npy`,
  - the upstream `model.staged_predict(X, prediction_type='RawFormulaVal',
    eval_period=k)` matrix for k in {1, 3}, each shaped `[n_stages, n_objects]`,
  - the EMPIRICALLY confirmed `stage_tree_counts` per schedule (R1): for each
    stage j, the tree count c_j such that `model.predict(X, ntree_end=c_j)`
    reproduces stage j — i.e. exactly which prefix upstream evaluates at each
    stage (does it start at eval_period or 0? is ntree_end always included?).

Run (from repo root):
    .venv/bin/python crates/cb-oracle/fixtures/staged_predict/gen_fixtures.py

No fabrication — every value comes straight from `staged_predict` / `predict`.
"""
import json
import os

import numpy as np
import catboost as cb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
INPUTS = os.path.join(ROOT, "crates", "cb-oracle", "fixtures", "inputs", "numeric_tiny")

N_TREES = 10
PERIODS = [1, 3]
# Tolerance for matching a staged row to a truncated `predict(ntree_end=c)` row.
MATCH_TOL = 1e-9


def fit_model():
    X = np.load(os.path.join(INPUTS, "X.npy")).astype(np.float64)
    y = np.load(os.path.join(INPUTS, "y.npy")).astype(np.float64)
    m = cb.CatBoostRegressor(
        loss_function="RMSE", boost_from_average=False, bootstrap_type="No",
        depth=2, iterations=N_TREES, l2_leaf_reg=3.0, leaf_estimation_iterations=1,
        leaf_estimation_method="Gradient", learning_rate=0.1, random_seed=0,
        random_strength=0, score_function="L2", thread_count=1, verbose=False,
    )
    m.fit(cb.Pool(X, y))
    return X, m


def confirm_stage_counts(model, X, staged_rows):
    """For each staged row, find the tree count c in 1..=N_TREES whose truncated
    `predict(ntree_end=c)` reproduces it (R1). Returns the confirmed counts."""
    truncated = {
        c: np.asarray(
            model.predict(X, prediction_type="RawFormulaVal", ntree_end=c),
            dtype=np.float64,
        )
        for c in range(1, N_TREES + 1)
    }
    counts = []
    for j, row in enumerate(staged_rows):
        row = np.asarray(row, dtype=np.float64).reshape(-1)
        matched = None
        for c in range(1, N_TREES + 1):
            if np.max(np.abs(truncated[c] - row)) <= MATCH_TOL:
                matched = c
                break
        if matched is None:
            raise SystemExit(
                f"stage {j}: no truncated predict(ntree_end=c) matched staged row"
            )
        counts.append(matched)
    return counts


def main():
    X, model = fit_model()
    model.save_model(os.path.join(HERE, "model.cbm"), format="cbm")

    config = {
        "catboost_version": "1.2.10",
        "input_dataset": "numeric_tiny",
        "scenario": "staged_predict",
        "loss_function": "RMSE",
        "thread_count": 1,
        "seed": 0,
        "n_trees": N_TREES,
        "note": (
            "SP-04 / R1: float-only oblivious CatBoostRegressor. "
            "staged_predict(X, prediction_type='RawFormulaVal', eval_period=k) "
            "frozen as [n_stages, n_objects]. stage_tree_counts[k] are the "
            "EMPIRICALLY confirmed tree counts each stage corresponds to "
            "(matched against predict(ntree_end=c))."
        ),
        "stage_tree_counts": {},
        "artifacts": ["model.cbm"],
    }

    for k in PERIODS:
        staged = list(
            model.staged_predict(X, prediction_type="RawFormulaVal", eval_period=k)
        )
        counts = confirm_stage_counts(model, X, staged)
        mat = np.asarray(
            [np.asarray(r, dtype=np.float64).reshape(-1) for r in staged],
            dtype=np.float64,
        )
        fname = f"staged_period{k}.npy"
        np.save(os.path.join(HERE, fname), mat)
        config["stage_tree_counts"][str(k)] = counts
        config["artifacts"].append(fname)
        print(f"eval_period={k}: n_stages={len(staged)} "
              f"stage_tree_counts={counts} shape={mat.shape}")

    with open(os.path.join(HERE, "config.json"), "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
