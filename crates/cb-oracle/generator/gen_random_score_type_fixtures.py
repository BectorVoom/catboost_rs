"""Freeze the `random_score_type` parity fixture (NormalWithModelSizeDecrease /
Gumbel).

`random_score_type` selects the distribution of the `random_strength`
split-score perturbation. Upstream differs in BOTH halves
(`greedy_tensor_search.cpp:861-866`, `rand_score.h:41-49`):

    NormalWithModelSizeDecrease
        stdev = random_strength * derivativesStDevFromZero * modelSizeMultiplier
        draw  = Val + NormalDistribution(rand, 0, stdev)

    Gumbel
        stdev = random_strength * derivativesStDevFromZero * 1.0
        draw  = Val + stdev * log(log(1.0 / rand.GenRandReal1()))

So Gumbel does NOT decay the perturbation as the model grows -- the model-size
multiplier is exactly the "...WithModelSizeDecrease" half of the other name.

The two also consume DIFFERENT amounts of RNG per candidate (the normal draw is
rejection sampling over PAIRS of uniforms; Gumbel takes one), which shifts the
whole downstream draw stream. That is why the separation is large rather than a
small change of noise scale.

# Discrimination

The parameter is INERT at `random_strength = 0` (no draw happens), so the
fixture pins that too -- it is what proves the default fit path is untouched.
Measured separation at `random_strength = 1`: ~0.68, stable across seeds.

Run:  python3 crates/cb-oracle/generator/gen_random_score_type_fixtures.py
"""

import json
import os

import numpy as np
from catboost import CatBoostRegressor

CATBOOST_VERSION = "1.2.10"

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "fixtures")
OUT_DIR = os.path.join(FIXTURES, "random_score_type")

SEED = 20260814
N_TRAIN = 400
N_FEATURES = 4
RANDOM_STRENGTH = 1.0

PARAMS = dict(
    iterations=5,
    depth=3,
    learning_rate=0.3,
    l2_leaf_reg=3.0,
    bootstrap_type="No",
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

TYPES = ["NormalWithModelSizeDecrease", "Gumbel"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(N_TRAIN, N_FEATURES)).astype(np.float64)
    y = (
        3.0 * x[:, 0] + 2.0 * x[:, 1] - x[:, 2] + 0.5 * x[:, 3]
    ).astype(np.float64)
    x_eval = np.ascontiguousarray(rng.normal(size=(12, N_FEATURES)), dtype=np.float64)

    np.save(os.path.join(OUT_DIR, "X.npy"), x)
    np.save(os.path.join(OUT_DIR, "y.npy"), y)
    np.save(os.path.join(OUT_DIR, "X_eval.npy"), x_eval)

    meta = {
        "scenario": "random_score_type",
        "catboost_version": CATBOOST_VERSION,
        "seed": SEED,
        "params": PARAMS,
        "random_strength": RANDOM_STRENGTH,
        "types": TYPES,
        "formulas": {
            "NormalWithModelSizeDecrease":
                "stdev = strength*dsdz*modelSizeMultiplier; draw = Val + Normal(0, stdev)",
            "Gumbel":
                "stdev = strength*dsdz*1.0; draw = Val + stdev*log(log(1/GenRandReal1()))",
        },
    }

    preds = {}
    for score_type in TYPES:
        m = CatBoostRegressor(
            random_score_type=score_type, random_strength=RANDOM_STRENGTH, **PARAMS
        )
        m.fit(x, y)
        p = np.asarray(m.predict(x_eval), dtype=np.float64)
        np.save(os.path.join(OUT_DIR, "preds_%s.npy" % score_type), p)
        preds[score_type] = p
        print("%-28s preds[:3] = %s" % (score_type, np.round(p[:3], 6)))

    # INERT at random_strength = 0 -- what proves the default path is untouched.
    zero = {}
    for score_type in TYPES:
        m = CatBoostRegressor(random_score_type=score_type, random_strength=0, **PARAMS)
        m.fit(x, y)
        zero[score_type] = np.asarray(m.predict(x_eval), dtype=np.float64)
    np.save(os.path.join(OUT_DIR, "preds_strength0.npy"), zero[TYPES[0]])
    inert = float(np.max(np.abs(zero[TYPES[0]] - zero[TYPES[1]])))
    meta["strength_zero_max_abs_diff_between_types"] = inert
    if inert != 0.0:
        raise AssertionError(
            "random_score_type changed the model at random_strength=0 (%g); it must "
            "be inert there" % inert
        )
    print("inert at random_strength=0 (max|diff| = %g)" % inert)

    sep = float(np.max(np.abs(preds[TYPES[0]] - preds[TYPES[1]])))
    meta["types_max_abs_diff"] = sep
    if sep == 0.0:
        raise AssertionError(
            "the two random_score_type values produced IDENTICAL predictions; the "
            "fixture cannot tell them apart"
        )
    print("Normal vs Gumbel max|diff| = %.6g" % sep)

    with open(os.path.join(OUT_DIR, "config.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    print("wrote", OUT_DIR)


if __name__ == "__main__":
    main()
