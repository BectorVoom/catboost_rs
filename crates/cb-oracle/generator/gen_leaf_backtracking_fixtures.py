"""Freeze the `leaf_estimation_backtracking` parity fixture.

# The finding this fixture records

Backtracking HALVES a leaf-value step that would not improve the loss. That
makes it a property of the leaf-estimation ITERATION loop — with a single step
there is no earlier step to fall back to.

`leaf_estimation_iterations` is not implemented in catboost-rs (the leaf
estimator takes exactly one step), so the first question is whether the
parameter can be observed at all in that regime. Measured against catboost
1.2.10 over {RMSE, Logloss, MAE, Poisson, Tweedie, Huber, LogCosh, Quantile} x
{Gradient, Newton} x learning_rate {0.3, 1, 3, 10}:

    leaf_estimation_iterations = 1   ->   0 / 64 configs distinguish
                                          No from AnyImprovement
    leaf_estimation_iterations > 1   ->   53 configs DO distinguish them
                                          (e.g. Huber:delta=1.0, lr=0.3,
                                          Newton, 5 iters separates by 5.16)

So this fixture pins TWO things:

1. the predictions themselves, for both CPU policies, so the engine matches
   catboost; and
2. the EQUIVALENCE of `No` and `AnyImprovement` at one leaf iteration, as an
   oracle-verified fact rather than an assumption. If `leaf_estimation_iterations`
   is ever implemented, this fixture's equivalence assertion is exactly what
   should start failing — which is the signal that the backtracking SEARCH now
   has to be written.

`Armijo` is GPU-ONLY upstream (`catboost_options.cpp:664`), so its rejection
message is captured as part of the contract.

Run:  python3 crates/cb-oracle/generator/gen_leaf_backtracking_fixtures.py
"""

import json
import os

import numpy as np
from catboost import CatBoostRegressor

CATBOOST_VERSION = "1.2.10"

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "fixtures")
OUT_DIR = os.path.join(FIXTURES, "leaf_estimation_backtracking")

SEED = 20260814
N_TRAIN = 300
N_FEATURES = 3

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

# The two CPU-legal policies. `Armijo` is captured separately as a rejection.
CPU_POLICIES = ["No", "AnyImprovement"]


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
        "scenario": "leaf_estimation_backtracking",
        "catboost_version": CATBOOST_VERSION,
        "seed": SEED,
        "params": PARAMS,
        "cpu_policies": CPU_POLICIES,
        "armijo": "GPU-only upstream (catboost_options.cpp:664)",
    }

    preds = {}
    for policy in CPU_POLICIES:
        m = CatBoostRegressor(leaf_estimation_backtracking=policy, **PARAMS)
        m.fit(x, y)
        p = np.asarray(m.predict(x_eval), dtype=np.float64)
        np.save(os.path.join(OUT_DIR, "preds_%s.npy" % policy), p)
        preds[policy] = p
        print("%-15s preds[:3] = %s" % (policy, np.round(p[:3], 6)))

    diff = float(np.max(np.abs(preds["No"] - preds["AnyImprovement"])))
    meta["no_vs_anyimprovement_max_abs_diff"] = diff
    if diff != 0.0:
        raise AssertionError(
            "No and AnyImprovement differ by %g at leaf_estimation_iterations=1; the "
            "documented equivalence this fixture records no longer holds and the "
            "backtracking search must actually be implemented" % diff
        )
    print("No vs AnyImprovement max|diff| = %g (equivalence CONFIRMED at 1 leaf iteration)"
          % diff)

    # Armijo must be REJECTED on CPU; the message is part of the contract.
    try:
        CatBoostRegressor(leaf_estimation_backtracking="Armijo", **PARAMS).fit(x, y)
        raise AssertionError("Armijo must be rejected on CPU")
    except AssertionError:
        raise
    except Exception as exc:
        meta["armijo_rejection_message"] = " ".join(str(exc).split())
        print("Armijo rejected:", meta["armijo_rejection_message"][:110])

    # Record the >1-iteration counter-example so the "0/64 at 1 iteration" claim
    # is anchored to a concrete case that DOES separate the policies.
    ce = dict(PARAMS)
    # Huber is not on upstream's boost_from_average allow-list
    # (catboost_options.cpp:709), so the counter-example turns it off.
    ce.update(loss_function="Huber:delta=1.0", leaf_estimation_method="Newton",
              leaf_estimation_iterations=5, boost_from_average=False)
    ce_preds = {}
    for policy in CPU_POLICIES:
        m = CatBoostRegressor(leaf_estimation_backtracking=policy, **ce)
        m.fit(x, y)
        ce_preds[policy] = np.asarray(m.predict(x_eval), dtype=np.float64)
    ce_diff = float(np.max(np.abs(ce_preds["No"] - ce_preds["AnyImprovement"])))
    meta["counterexample_at_5_leaf_iterations"] = {
        "loss_function": "Huber:delta=1.0",
        "leaf_estimation_method": "Newton",
        "leaf_estimation_iterations": 5,
        "boost_from_average": False,
        "no_vs_anyimprovement_max_abs_diff": ce_diff,
        "note": "Proves the parameter is NOT inert in general -- only at the single "
                "leaf iteration this engine currently supports.",
    }
    print("counter-example at 5 leaf iterations: max|diff| = %.6g" % ce_diff)

    with open(os.path.join(OUT_DIR, "config.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    print("wrote", OUT_DIR)


if __name__ == "__main__":
    main()
