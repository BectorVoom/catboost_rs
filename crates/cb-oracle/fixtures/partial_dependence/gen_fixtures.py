#!/usr/bin/env python3
"""Offline fixture generator for FSTR-03 partial dependence (PDP-02..04).

Generates, from catboost==1.2.10, the upstream partial-dependence ground truth
for a numeric-only regression model trained on `inputs/numeric_tiny`:

  - model.cbm / model.json                 : the pinned model (loaded by Rust)
  - pdp_single_values.npy                  : upstream 1-D PD for `single_feature`
  - pdp_pair_values.npy                    : upstream 2-D PD for `pair_features`,
                                             row-major C-order (f1 outer, f2 inner)
  - config.json                            : params, feature choices, bin counts

Oracle truth is `CatBoost.plot_partial_dependence(pool, features, plot=False)[0]`
== `_calc_partial_dependence(...)` — the model's own averaged per-BIN prediction,
NOT a hand-written averaging loop (core.py:4033-4055). Upstream computes ONE value
per BIN (n_borders+1 bins); a feature with 0 borders is rejected upstream
("not used in model"), so we only use features that the model actually split on.

Run (from repo root), with catboost==1.2.10 available:
    <py3.12-venv>/bin/python crates/cb-oracle/fixtures/partial_dependence/gen_fixtures.py

Pinned seed, thread_count=1, bootstrap_type="No". No fabrication.
"""
import json
import os

import numpy as np
import catboost as cb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
INPUTS = os.path.join(ROOT, "crates", "cb-oracle", "fixtures", "inputs", "numeric_tiny")

# Feature 3 has 3 borders -> 4 bins (the richest 1-D curve); features 0 & 3 are
# both used (non-empty borders) -> a valid 2-D pair. Feature 2 has 0 borders and
# is rejected by upstream, so it is deliberately avoided.
SINGLE_FEATURE = 3
PAIR_FEATURES = [0, 3]

ISOLATING = dict(
    boost_from_average=False, bootstrap_type="No", depth=2, iterations=5,
    l2_leaf_reg=3.0, leaf_estimation_iterations=1, leaf_estimation_method="Gradient",
    learning_rate=0.1, random_seed=0, random_strength=0, score_function="L2",
    thread_count=1, verbose=False,
)


def main():
    X = np.load(os.path.join(INPUTS, "X.npy")).astype(np.float64)
    y = np.load(os.path.join(INPUTS, "y.npy"))
    pool = cb.Pool(X, y)

    m = cb.CatBoostRegressor(**ISOLATING)
    m.fit(pool)
    m.save_model(os.path.join(HERE, "model.cbm"), format="cbm")
    m.save_model(os.path.join(HERE, "model.json"), format="json")

    borders = {i: list(map(float, m._get_borders()[i])) for i in range(X.shape[1])}

    # --- 1-D: one PD value per bin of SINGLE_FEATURE ---
    single_vals, _ = m.plot_partial_dependence(pool, features=[SINGLE_FEATURE], plot=False)
    single_vals = np.asarray(single_vals, dtype=np.float64)
    np.save(os.path.join(HERE, "pdp_single_values.npy"), single_vals)

    # --- 2-D: PD surface over PAIR_FEATURES, C-order row-major (f1 outer, f2 inner) ---
    pair_vals, _ = m.plot_partial_dependence(pool, features=PAIR_FEATURES, plot=False)
    pair_vals = np.asarray(pair_vals, dtype=np.float64)  # shape (n_b0+1, n_b1+1)
    pair_flat = pair_vals.reshape(-1, order="C")
    np.save(os.path.join(HERE, "pdp_pair_values.npy"), pair_flat)

    config = {
        "catboost_version": "1.2.10",
        "input_dataset": "numeric_tiny",
        "scenario": "partial_dependence",
        "thread_count": 1,
        "random_seed": 0,
        "single_feature": SINGLE_FEATURE,
        "pair_features": PAIR_FEATURES,
        "single_n_bins": int(single_vals.shape[0]),
        "pair_shape_row_major": [int(pair_vals.shape[0]), int(pair_vals.shape[1])],
        "pair_axis_to_feature": {"axis0_outer": PAIR_FEATURES[0], "axis1_inner": PAIR_FEATURES[1]},
        "borders": {str(k): v for k, v in borders.items()},
        "grid_convention": (
            "per-BIN: n_borders+1 bins. Rust grid representative for bin i = "
            "[b0-1.0, (b_{i-1}+b_i)/2 ..., b_last+1.0]; any interior value maps to "
            "the same bin so the averaged RawFormulaVal equals upstream _calc_partial_dependence."
        ),
        "note": (
            "FSTR-03 partial dependence (PDP-02..04). Values are upstream "
            "plot_partial_dependence(pool, features, plot=False)[0] == "
            "_calc_partial_dependence (core.py:4041). ONE value per bin. "
            "pdp_pair_values.npy is C-order row-major (feature %d outer, %d inner)."
            % (PAIR_FEATURES[0], PAIR_FEATURES[1])
        ),
        "artifacts": [
            "model.cbm", "model.json",
            "pdp_single_values.npy", "pdp_pair_values.npy",
        ],
    }
    with open(os.path.join(HERE, "config.json"), "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    print("single_feature", SINGLE_FEATURE, "->", single_vals.shape, single_vals)
    print("pair_features", PAIR_FEATURES, "-> shape", pair_vals.shape, "flat len", pair_flat.shape[0])


if __name__ == "__main__":
    main()
