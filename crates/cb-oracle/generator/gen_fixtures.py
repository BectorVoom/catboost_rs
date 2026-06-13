#!/usr/bin/env python3
"""Generate per-stage expected-OUTPUT oracle fixtures (regression_skeleton).

BUILD-TIME tool (D-12): runs OUTSIDE CI, writes committed frozen fixtures. CI
only READS the committed artifacts.

Loads the frozen `numeric_tiny` INPUT corpus (produced by `gen_inputs.py`),
trains a pinned `CatBoostRegressor`, and extracts the per-stage oracle values the
Rust harness compares against at <= 1e-5 (INFRA-04):

    get_borders()            -> borders.npy        (flattened f64, layout below)
    save_model(format=json)  -> model.json         (splits + leaf_values, INFRA-04)
    staged_predict()         -> staged.npy         (n_iterations x n_rows, flat f64)
    predict()                -> predictions.npy    (n_rows f64)

Determinism (Pitfall 2 / T-01-07): thread_count=1 and a fixed random_seed make
the summation order reproducible; the exact params + seed are recorded in
fixtures/regression_skeleton/config.json so the baseline is auditable.

borders.npy layout: borders for feature 0, then feature 1, ... concatenated in
ascending feature-index order. borders_per_feature.npy records the count for each
feature (f64-encoded integer counts) so a reader can split the flat vector back
into per-feature border lists.

Run (after gen_inputs.py):
    .venv/bin/python gen_fixtures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
INPUTS = FIXTURES / "inputs"
SCENARIO = FIXTURES / "regression_skeleton"

CATBOOST_VERSION = "1.2.10"
SEED = 0
PARAMS = {
    "iterations": 10,
    "learning_rate": 0.1,
    "depth": 4,
    "random_seed": SEED,
    "thread_count": 1,  # Pitfall 2: pin to 1 for deterministic summation order.
    "verbose": False,
}


def _assert_f64(arr: np.ndarray, name: str) -> np.ndarray:
    if arr.dtype != np.float64:
        raise AssertionError(f"{name} must be np.float64, got {arr.dtype}")
    return arr


def main() -> None:
    # Load the frozen numeric_tiny input corpus.
    x = np.load(INPUTS / "numeric_tiny" / "X.npy")
    y = np.load(INPUTS / "numeric_tiny" / "y.npy")

    model = CatBoostRegressor(**PARAMS)
    model.fit(x, y)

    SCENARIO.mkdir(parents=True, exist_ok=True)

    # --- Stage: Borders -----------------------------------------------------
    # get_borders() -> dict {feature_index: [border, ...]} in 1.2.10.
    borders_dict = model.get_borders()
    feature_indices = sorted(borders_dict.keys())
    flat_borders: list[float] = []
    counts: list[float] = []
    for fi in feature_indices:
        feat_borders = list(borders_dict[fi])
        flat_borders.extend(float(b) for b in feat_borders)
        counts.append(float(len(feat_borders)))
    borders_arr = _assert_f64(np.asarray(flat_borders, dtype=np.float64), "borders")
    counts_arr = _assert_f64(np.asarray(counts, dtype=np.float64), "borders_per_feature")
    np.save(SCENARIO / "borders.npy", borders_arr, allow_pickle=False)
    np.save(SCENARIO / "borders_per_feature.npy", counts_arr, allow_pickle=False)

    # --- Stage: Splits + LeafValues (model.json) ----------------------------
    model.save_model(str(SCENARIO / "model.json"), format="json")

    # --- Stage: StagedApprox ------------------------------------------------
    staged = [np.asarray(p, dtype=np.float64) for p in model.staged_predict(x)]
    staged_flat = _assert_f64(
        np.concatenate([s.ravel() for s in staged]).astype(np.float64), "staged"
    )
    np.save(SCENARIO / "staged.npy", staged_flat, allow_pickle=False)

    # --- Stage: Predictions -------------------------------------------------
    predictions = _assert_f64(np.asarray(model.predict(x), dtype=np.float64), "predictions")
    np.save(SCENARIO / "predictions.npy", predictions, allow_pickle=False)

    # --- config.json (hybrid fixture metadata, D-09) ------------------------
    config = {
        "scenario": "regression_skeleton",
        "seed": SEED,
        "catboost_version": CATBOOST_VERSION,
        "thread_count": 1,
        "input_dataset": "numeric_tiny",
        "params": PARAMS,
        "n_rows": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "n_iterations": len(staged),
        "borders_feature_indices": feature_indices,
        "stages": ["Borders", "Splits", "LeafValues", "StagedApprox", "Predictions"],
        "borders_layout": (
            "flat f64: feature 0 borders, then feature 1, ... ; "
            "borders_per_feature.npy gives the per-feature counts"
        ),
        "staged_layout": "flat f64: stage 0 (n_rows), then stage 1, ... ; n_iterations stages",
    }
    with (SCENARIO / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(f"Wrote regression_skeleton oracle fixtures under {SCENARIO}")
    print(
        f"  borders={borders_arr.shape}, staged={staged_flat.shape}, "
        f"predictions={predictions.shape}"
    )


if __name__ == "__main__":
    main()
