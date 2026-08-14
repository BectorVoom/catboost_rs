"""Freeze the `leaf_estimation_iterations` parity fixture.

Upstream's `CalcApproxDeltaSimple` runs the leaf solve N times per tree,
ACCUMULATING the per-leaf delta and RECOMPUTING the derivatives at
`approx + accumulated_delta` before each further step. One iteration is the
single-step solve this port already had.

This fixture is deliberately generated for BOTH backtracking policies at N > 1,
because that is the regime where they finally diverge (at N = 1 they are
provably identical -- see `leaf_estimation_backtracking/`). So it pins:

  1. the multi-step estimator itself, at `backtracking=No` (pure accumulation,
     no step shrinking); and
  2. the backtracking search, as the DELTA between `No` and `AnyImprovement`
     at the same N.

Loss choice: Poisson with the Newton leaf, reached by elimination.

  * RMSE + Gradient -- the leaf solve is already the exact optimum, so extra
    steps add almost nothing and N=1 vs N=5 barely separates.
  * Logloss + Newton -- separates N=1 from N=5 (by 1.69), but the Newton step
    always improves, so AnyImprovement never fires and the two policies stayed
    BYTE-IDENTICAL even at N=5.
  * Huber + Newton -- separates the policies (by 7.12), but AnyImprovement
    collapses to the all-zero model: backtracking rejects every step. An
    all-zero expectation is a near-VACUOUS oracle, since an implementation that
    simply refused every step would match it.
  * Poisson + Newton -- separates the policies by 0.251 with BOTH models
    non-trivial (max |AnyImprovement| = 1.76, max |No| = 1.79), so some steps
    are accepted and some shrunk. That is the regime that actually exercises the
    search.

Run:  python3 crates/cb-oracle/generator/gen_leaf_iterations_fixtures.py
"""

import json
import os

import numpy as np
from catboost import CatBoostRegressor

CATBOOST_VERSION = "1.2.10"

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "fixtures")
OUT_DIR = os.path.join(FIXTURES, "leaf_estimation_iterations")

SEED = 20260814
N_TRAIN = 300
N_FEATURES = 3

PARAMS = dict(
    # See the module doc for why this loss and leaf method: it is the only
    # combination tried that both separates the two backtracking policies AND
    # leaves both models non-degenerate.
    loss_function="Poisson",
    iterations=5,
    depth=3,
    learning_rate=0.3,
    l2_leaf_reg=3.0,
    bootstrap_type="No",
    random_strength=0,
    score_function="L2",
    leaf_estimation_method="Newton",
    random_seed=0,
    thread_count=1,
    verbose=False,
    boost_from_average=False,
    border_count=32,
    grow_policy="SymmetricTree",
)

ITER_COUNTS = [1, 2, 5]
POLICIES = ["No", "AnyImprovement"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(N_TRAIN, N_FEATURES)).astype(np.float64)
    # Poisson needs a POSITIVE target (exp-link, positive domain).
    y = (np.abs(3.0 * x[:, 0] + 2.0 * x[:, 1] - x[:, 2]) + 0.1).astype(np.float64)
    x_eval = np.ascontiguousarray(rng.normal(size=(16, N_FEATURES)), dtype=np.float64)

    np.save(os.path.join(OUT_DIR, "X.npy"), x)
    np.save(os.path.join(OUT_DIR, "y.npy"), y)
    np.save(os.path.join(OUT_DIR, "X_eval.npy"), x_eval)

    meta = {
        "scenario": "leaf_estimation_iterations",
        "catboost_version": CATBOOST_VERSION,
        "seed": SEED,
        "params": PARAMS,
        "iteration_counts": ITER_COUNTS,
        "backtracking_policies": POLICIES,
        "prediction_type": "RawFormulaVal",
    }

    preds = {}
    for policy in POLICIES:
        for n_iter in ITER_COUNTS:
            m = CatBoostRegressor(
                leaf_estimation_iterations=n_iter,
                leaf_estimation_backtracking=policy,
                **PARAMS,
            )
            m.fit(x, y)
            p = np.asarray(
                m.predict(x_eval, prediction_type="RawFormulaVal"), dtype=np.float64
            )
            np.save(os.path.join(OUT_DIR, "preds_%s_%d.npy" % (policy, n_iter)), p)
            preds[(policy, n_iter)] = p
            print("%-15s iters=%-2d preds[:3] = %s" % (policy, n_iter, np.round(p[:3], 6)))

    # (1) More steps must CHANGE the model, or the fixture cannot detect a
    #     single-step implementation.
    for policy in POLICIES:
        d = float(np.max(np.abs(preds[(policy, 5)] - preds[(policy, 1)])))
        meta["%s_iters5_vs_iters1_max_abs_diff" % policy] = d
        if d == 0.0:
            raise AssertionError(
                "%s: 5 leaf iterations is indistinguishable from 1; the fixture "
                "would pass for an implementation that ignores the parameter" % policy
            )
        print("%-15s |iters5 - iters1| = %.6g" % (policy, d))

    # (2) At N=1 the two policies must still coincide (the previously frozen fact).
    same_at_1 = float(np.max(np.abs(preds[("No", 1)] - preds[("AnyImprovement", 1)])))
    meta["policies_agree_at_one_iteration_max_abs_diff"] = same_at_1
    if same_at_1 != 0.0:
        raise AssertionError(
            "No and AnyImprovement differ at 1 leaf iteration (%g); that contradicts "
            "the leaf_estimation_backtracking fixture" % same_at_1
        )

    # (3) At N>1 they must DIVERGE, which is what makes the backtracking search
    #     testable at all.
    sep = float(np.max(np.abs(preds[("No", 5)] - preds[("AnyImprovement", 5)])))
    meta["policies_differ_at_five_iterations_max_abs_diff"] = sep
    print("policies at iters=5 differ by %.6g" % sep)
    if sep == 0.0:
        raise AssertionError(
            "No and AnyImprovement still coincide at 5 leaf iterations; this fixture "
            "cannot exercise the backtracking search"
        )

    # (4) AnyImprovement must not be the DEGENERATE all-zero model. If every step
    #     is rejected the expectation is all zeros, which an implementation that
    #     simply refuses every step would also produce -- a near-vacuous oracle.
    any_mag = float(np.max(np.abs(preds[("AnyImprovement", 5)])))
    no_mag = float(np.max(np.abs(preds[("No", 5)])))
    meta["anyimprovement_iters5_max_abs_prediction"] = any_mag
    meta["no_iters5_max_abs_prediction"] = no_mag
    if any_mag <= 1e-6:
        raise AssertionError(
            "AnyImprovement collapsed to the all-zero model (max |pred| = %g): every "
            "step was rejected, so this fixture cannot distinguish a real backtracking "
            "search from one that refuses everything" % any_mag
        )
    print("non-degenerate: max|AnyImprovement| = %.4g, max|No| = %.4g" % (any_mag, no_mag))

    with open(os.path.join(OUT_DIR, "config.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    print("wrote", OUT_DIR)


if __name__ == "__main__":
    main()
