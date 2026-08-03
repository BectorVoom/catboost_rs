"""Generate the counter_calc_method eval-set discrimination fixture (E23, SPEC-CTRT-17 / A6).

FROZEN GENERATOR — the committed artifacts under this directory are the GROUND
TRUTH the repo's `counter_calc_method` threading is compared against at <=1e-5.
CI does NOT run this script (no `catboost` install in CI) and never regenerates
these fixtures.

# Why this fixture exists — and why it MUST carry an eval set

`counter_calc_method` is UNOBSERVABLE without an eval set: measured
`maxdiff = 0.000e+00` learn-only vs `4.010e-01` with a 40-row eval set
(research §B, probe6/probe7). A learn-only test passes trivially and proves
nothing — it is FORBIDDEN by the plan. This fixture trains the SAME corpus
twice, once at `counter_calc_method="Full"` and once at `"SkipTest"`, both with
the SAME eval set, and asserts the two prediction vectors genuinely differ (the
discriminator guard) before freezing BOTH.

# Reproducibility caveat (load-bearing — do not "fix" by re-running this file)

Categorical-only feature matrix (zero float columns) so upstream
float-quantization nondeterminism is structurally excluded, the same mitigation
every other `ctr_*` fixture uses. The committed artifacts ARE the fixtures.

# Pinned recipe

  - `catboost==1.2.10`, `numpy.random.RandomState(0)`.
  - 60 learn rows + 40 eval rows (the measured probe scale), TWO categorical
    columns (cardinality 6 and 5), ZERO float columns.
  - `one_hot_max_size=1`; `simple_ctr=["Counter:Prior=0.5"]`,
    `combinations_ctr=[]`, `max_ctr_complexity=1`.
  - The eval rows deliberately skew toward high category codes so the eval
    tally genuinely moves the Counter totals (and the eval set contains codes
    the learn slice also has, plus the tally shift the guard demands).
  - The low-level `CatBoost(params)` API with `eval_set=Pool(...)`.
"""

import json
import os
import subprocess
import sys

import numpy as np
from catboost import CatBoost, Pool

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.dirname(HERE)
SCENARIO = "ctr_counter_full_eval"

N_LEARN = 60
N_EVAL = 40
CARD0 = 6
CARD1 = 5


def params(method):
    return {
        "loss_function": "Logloss",
        "iterations": 10,
        "depth": 2,
        "learning_rate": 0.1,
        "l2_leaf_reg": 3.0,
        "boosting_type": "Plain",
        "one_hot_max_size": 1,
        "max_ctr_complexity": 1,
        "simple_ctr": ["Counter:Prior=0.5"],
        "combinations_ctr": [],
        "permutation_count": 1,
        "fold_len_multiplier": 2.0,
        "counter_calc_method": method,
        "leaf_estimation_method": "Gradient",
        "leaf_estimation_iterations": 1,
        "bootstrap_type": "No",
        # PIN use_best_model: with an eval_set, catboost's raw dict-API default
        # flips to True and TRUNCATES the model to the best eval iteration
        # (observed: the un-pinned SkipTest run kept 1 of 10 trees) — the
        # classic unpinned-default trap (cv-orch01). Both settings must train
        # the full 10 trees for the discriminator to isolate counter_calc_method.
        "use_best_model": False,
        "random_strength": 0,
        "random_seed": 0,
        "thread_count": 1,
        "boost_from_average": False,
        "verbose": False,
    }


def main():
    rng = np.random.RandomState(0)

    c0 = rng.randint(0, CARD0, size=N_LEARN).astype(np.int32)
    c1 = rng.randint(0, CARD1, size=N_LEARN).astype(np.int32)
    x_cat = np.stack([c0, c1], axis=1).astype(np.int32)
    logit = 1.2 * (c0 % 2) - 0.9 * (c1 % 3) + 0.3 * (c0 // 3)
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.rand(N_LEARN) < prob).astype(np.int32)

    # Eval rows skewed toward the upper category range so the Full tally
    # genuinely shifts the per-bucket Counter totals.
    e0 = rng.randint(CARD0 // 2, CARD0, size=N_EVAL).astype(np.int32)
    e1 = rng.randint(CARD1 // 2, CARD1, size=N_EVAL).astype(np.int32)
    x_eval = np.stack([e0, e1], axis=1).astype(np.int32)
    elogit = 1.2 * (e0 % 2) - 0.9 * (e1 % 3) + 0.3 * (e0 // 3)
    eprob = 1.0 / (1.0 + np.exp(-elogit))
    y_eval = (rng.rand(N_EVAL) < eprob).astype(np.int32)

    if len(np.unique(y)) < 2:
        sys.exit("degenerate learn target: only one class present")

    learn_pool = Pool(x_cat, label=y, cat_features=[0, 1])
    eval_pool = Pool(x_eval, label=y_eval, cat_features=[0, 1])

    results = {}
    for method in ("Full", "SkipTest"):
        model = CatBoost(params(method))
        model.fit(learn_pool, eval_set=eval_pool)
        preds = model.predict(learn_pool, prediction_type="RawFormulaVal")
        mj_path = os.path.join(HERE, f"model_{method.lower()}.json")
        model.save_model(mj_path, format="json")
        with open(mj_path) as fh:
            results[method] = (preds, json.load(fh))

    pred_full, mj_full = results["Full"]
    pred_skiptest, _mj_skip = results["SkipTest"]

    # --- MANDATORY anti-false-pass guard: THE DISCRIMINATOR itself -----------
    maxdiff = float(np.max(np.abs(pred_full - pred_skiptest)))
    assert maxdiff > 1e-3, (
        f"counter_calc_method is UNOBSERVABLE on this fixture (maxdiff={maxdiff:.3e}). "
        "A fixture where Full and SkipTest agree cannot test SPEC-CTRT-17. "
        "Research measured 4.010e-01 with a 40-row eval set — widen the eval set or "
        "strengthen the categorical signal. DO NOT weaken this assertion."
    )
    ctrs_full = mj_full["features_info"].get("ctrs", [])
    assert any(c["ctr_type"] == "Counter" for c in ctrs_full), (
        "no Counter CTR descriptor in model_full.json — the fixture is vacuous"
    )
    assert pred_full.std() > 1e-6 and pred_skiptest.std() > 1e-6

    np.save(os.path.join(HERE, "X_cat.npy"), x_cat)
    np.save(os.path.join(HERE, "y.npy"), y.astype(np.float64))
    np.save(os.path.join(HERE, "X_cat_eval.npy"), x_eval)
    np.save(os.path.join(HERE, "y_eval.npy"), y_eval.astype(np.float64))
    np.save(os.path.join(HERE, "predictions_full.npy"), pred_full.astype(np.float64))
    np.save(
        os.path.join(HERE, "predictions_skiptest.npy"), pred_skiptest.astype(np.float64)
    )

    config = {
        "scenario": SCENARIO,
        "requirement": "SPEC-CTRT-17",
        "seed": 0,
        "catboost_version": "1.2.10",
        "thread_count": 1,
        "n_learn": N_LEARN,
        "n_eval": N_EVAL,
        "description": (
            "counter_calc_method eval-set discrimination (E23 / A6). Categorical-ONLY, "
            "60 learn + 40 eval rows, simple Counter CTR. Trained TWICE — Full vs "
            "SkipTest, same eval set — and the guard asserts the two prediction "
            "vectors differ by > 1e-3, so a threading bug that makes the repo "
            "compute either setting for both cannot pass both gates. "
            "NEVER regenerated in CI."
        ),
        "params_full": params("Full"),
        "params_skiptest": params("SkipTest"),
        "observed_full_vs_skiptest_maxdiff": maxdiff,
        "npy_schema": {
            "X_cat.npy": "[60,2] int32 — learn categorical codes",
            "y.npy": "[60] f64 — learn binclf label",
            "X_cat_eval.npy": "[40,2] int32 — eval categorical codes",
            "y_eval.npy": "[40] f64 — eval binclf label",
            "predictions_full.npy": "[60] f64 — RawFormulaVal under Full (<=1e-5 gate)",
            "predictions_skiptest.npy": "[60] f64 — RawFormulaVal under SkipTest (<=1e-5 gate)",
        },
        "note": (
            "FROZEN. Generated once by this directory's own gen_fixtures.py and "
            "NEVER regenerated in CI. Regenerating invalidates the <=1e-5 gates."
        ),
    }
    with open(os.path.join(HERE, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)
        fh.write("\n")

    out = subprocess.run(
        ["git", "status", "--porcelain", FIXTURES],
        capture_output=True,
        text=True,
        cwd=FIXTURES,
    ).stdout
    offenders = [
        line for line in out.splitlines() if line.strip() and SCENARIO not in line
    ]
    if offenders:
        print("corpus contamination — this generator touched paths outside its scenario:")
        for line in offenders:
            print("   ", line)
        sys.exit(1)

    print(f"wrote {SCENARIO}: Full-vs-SkipTest maxdiff = {maxdiff:.3e}")


if __name__ == "__main__":
    main()
