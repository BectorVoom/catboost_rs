"""Freeze the `model_shrink_rate` / `model_shrink_mode` parity fixture.

# What upstream does

Before growing each new tree, catboost multiplies the ENTIRE accumulated model —
the bias AND every already-grown tree's leaf values — by a factor below 1. That
also rescales the running approximant the next tree's gradients come from, so it
is a training-dynamics change rather than a post-hoc rescale. The model-level
`scale` stays 1: the shrinkage is baked into the leaves and the bias.

The first tree is never shrunk, so with `iterations = 5` there are FOUR
applications. Reading catboost's own saved leaf values back at `rate = 0.1`,
`learning_rate = 0.3` gives the two multipliers exactly:

    Constant    tree0 leaves scaled by 0.88529 = 0.97^4
                                                 = (1 - rate*lr)^4
    Decreasing  tree0 leaves scaled by 0.80583 = 0.9 * 0.95 * 0.96667 * 0.975
                                                 = prod_i (1 - rate/i), i=1..4

# Discrimination

`rate = 0.0` must be INERT (byte-identical to a fit that never mentions the
parameter), and the two modes must differ from each other and from the
unshrunk fit. The generator asserts all three, so the fixture cannot pass for an
implementation that ignores either knob.

Run:  python3 crates/cb-oracle/generator/gen_model_shrink_fixtures.py
"""

import json
import os

import numpy as np
from catboost import CatBoostRegressor

CATBOOST_VERSION = "1.2.10"

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "fixtures")
OUT_DIR = os.path.join(FIXTURES, "model_shrink")

SEED = 20260814
N_TRAIN = 300
N_FEATURES = 3
SHRINK_RATE = 0.1

PARAMS = dict(
    iterations=5,
    depth=3,
    learning_rate=0.3,
    l2_leaf_reg=3.0,
    bootstrap_type="No",
    random_strength=0,
    leaf_estimation_iterations=1,
    score_function="L2",
    leaf_estimation_method="Gradient",
    random_seed=0,
    thread_count=1,
    verbose=False,
    boost_from_average=True,
    border_count=32,
    grow_policy="SymmetricTree",
)

MODES = ["Constant", "Decreasing"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(N_TRAIN, N_FEATURES)).astype(np.float64)
    y = (3.0 * x[:, 0] + 2.0 * x[:, 1] - x[:, 2]).astype(np.float64)
    x_eval = np.ascontiguousarray(rng.normal(size=(16, N_FEATURES)), dtype=np.float64)

    np.save(os.path.join(OUT_DIR, "X.npy"), x)
    np.save(os.path.join(OUT_DIR, "y.npy"), y)
    np.save(os.path.join(OUT_DIR, "X_eval.npy"), x_eval)

    meta = {
        "scenario": "model_shrink",
        "catboost_version": CATBOOST_VERSION,
        "seed": SEED,
        "params": PARAMS,
        "model_shrink_rate": SHRINK_RATE,
        "modes": MODES,
        "multipliers": {
            "Constant": "1 - model_shrink_rate * learning_rate, every iteration",
            "Decreasing": "1 - model_shrink_rate / i at 1-based shrink step i",
        },
    }

    # Baseline: no shrinkage at all.
    base = CatBoostRegressor(**PARAMS)
    base.fit(x, y)
    p_base = np.asarray(base.predict(x_eval), dtype=np.float64)
    np.save(os.path.join(OUT_DIR, "preds_none.npy"), p_base)

    # rate = 0 must be INERT.
    zero = CatBoostRegressor(model_shrink_rate=0.0, model_shrink_mode="Constant", **PARAMS)
    zero.fit(x, y)
    p_zero = np.asarray(zero.predict(x_eval), dtype=np.float64)
    inert_diff = float(np.max(np.abs(p_zero - p_base)))
    meta["rate_zero_vs_unset_max_abs_diff"] = inert_diff
    if inert_diff != 0.0:
        raise AssertionError(
            "model_shrink_rate=0 changed the model by %g; it must be inert" % inert_diff
        )
    print("rate=0 is inert (max|diff| = %g)" % inert_diff)

    preds = {}
    for mode in MODES:
        m = CatBoostRegressor(
            model_shrink_rate=SHRINK_RATE, model_shrink_mode=mode, **PARAMS
        )
        m.fit(x, y)
        p = np.asarray(m.predict(x_eval), dtype=np.float64)
        np.save(os.path.join(OUT_DIR, "preds_%s.npy" % mode), p)
        preds[mode] = p
        print("%-11s max|diff vs unshrunk| = %.6g" % (mode, float(np.max(np.abs(p - p_base)))))

    sep = float(np.max(np.abs(preds["Constant"] - preds["Decreasing"])))
    meta["constant_vs_decreasing_max_abs_diff"] = sep
    if sep == 0.0:
        raise AssertionError(
            "Constant and Decreasing produced IDENTICAL predictions: the fixture "
            "cannot tell the two modes apart"
        )
    for mode in MODES:
        d = float(np.max(np.abs(preds[mode] - p_base)))
        meta["%s_vs_unshrunk_max_abs_diff" % mode] = d
        if d == 0.0:
            raise AssertionError(
                "%s is indistinguishable from an unshrunk fit; the fixture would "
                "pass for an implementation that ignores model_shrink_rate" % mode
            )
    print("Constant vs Decreasing max|diff| = %.6g" % sep)

    with open(os.path.join(OUT_DIR, "config.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    print("wrote", OUT_DIR)


if __name__ == "__main__":
    main()
