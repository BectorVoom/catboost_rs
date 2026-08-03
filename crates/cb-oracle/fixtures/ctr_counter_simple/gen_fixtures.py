"""Generate the Counter-CTR end-to-end parity fixture (E12, SPEC-CTRT-08 / A3).

FROZEN GENERATOR — the committed `X_cat.npy` / `y.npy` / `model.json` /
`predictions.npy` / `config.json` under this directory are the GROUND TRUTH the
repo's Counter-CTR training path is compared against at <=1e-5. CI does NOT run
this script (no `catboost` install in CI) and never regenerates these fixtures.

# Reproducibility caveat (load-bearing — do not "fix" by re-running this file)

CatBoost's quantization/border step has a documented source of run-to-run
nondeterminism independent of the exposed `random_seed` (observed even at
`thread_count=1`) — see `crates/cb-oracle/fixtures/ctr_load/gen_fixtures.py`.
This fixture sidesteps the FLOAT half of that problem structurally by using a
CATEGORICAL-ONLY feature matrix (no float columns => no float-border selection
at all), the same mitigation `tensor_ctr_e2e` uses. The committed artifacts ARE
the fixtures; this script records the pinned recipe for provenance.

# Why this fixture exists

`simple_ctr` was inert before this plan: the engine trained Borders CTRs no
matter what CTR type the caller configured. This fixture pins the Counter type
end to end against real upstream CatBoost, so a regression that silently falls
back to Borders is caught numerically rather than structurally.

# Pinned recipe

  - `catboost==1.2.10`, `numpy.random.RandomState(0)`.
  - 30 rows, TWO categorical columns (cardinality 6 and 5), ZERO float columns.
  - `one_hot_max_size=1`, so both columns route to the CTR path rather than
    one-hot (`route_categorical(card, 1) == Ctr`).
  - `simple_ctr=["Counter:Prior=0.5"]`, `combinations_ctr=[]`,
    `max_ctr_complexity=1` — simple Counter CTRs only.
  - The target is CAT-DRIVEN so upstream actually selects CTR splits; a weak or
    absent categorical signal yields ZERO CTR splits and the fixture would pass
    trivially. The anti-false-pass guard below asserts that outcome away.
  - The low-level `CatBoost(params)` API is used, NOT
    `CatBoostClassifier(**kwargs)`, so non-sklearn keys such as `simple_ctr`
    are honored.
"""

import json
import os
import subprocess
import sys

import numpy as np
from catboost import CatBoost, Pool

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.dirname(HERE)
SCENARIO = "ctr_counter_simple"

N_ROWS = 30
CARD0 = 6
CARD1 = 5

PARAMS = {
    "loss_function": "Logloss",
    "iterations": 5,
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
    "counter_calc_method": "SkipTest",
    "leaf_estimation_method": "Gradient",
    "leaf_estimation_iterations": 1,
    "bootstrap_type": "No",
    "random_strength": 0,
    "random_seed": 0,
    "thread_count": 1,
    "boost_from_average": False,
    "verbose": False,
}


def main():
    rng = np.random.RandomState(0)

    # Categorical-only design matrix. int32 on disk; the Rust side stringifies
    # via `cb_data::stringify_int_category`, the A4 plain-integer form CatBoost
    # hashed here.
    c0 = rng.randint(0, CARD0, size=N_ROWS).astype(np.int32)
    c1 = rng.randint(0, CARD1, size=N_ROWS).astype(np.int32)
    x_cat = np.stack([c0, c1], axis=1).astype(np.int32)

    # CAT-DRIVEN target: the label depends on the categorical columns, so
    # upstream genuinely selects CTR splits.
    logit = 1.2 * (c0 % 2) - 0.9 * (c1 % 3) + 0.3 * (c0 // 3)
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.rand(N_ROWS) < prob).astype(np.int32)

    if len(np.unique(y)) < 2:
        sys.exit("degenerate target: only one class present")

    pool = Pool(x_cat, label=y, cat_features=[0, 1])
    model = CatBoost(PARAMS)
    model.fit(pool)

    predictions = model.predict(pool, prediction_type="RawFormulaVal")

    model_json_path = os.path.join(HERE, "model.json")
    model.save_model(model_json_path, format="json")
    with open(model_json_path) as fh:
        model_json = json.load(fh)

    # --- MANDATORY anti-false-pass guard --------------------------------------
    # Without this, "the model trained" is satisfiable by a configuration that
    # produced ZERO Counter splits, and both sides would agree trivially.
    ctrs = model_json["features_info"].get("ctrs", [])
    assert any(c["ctr_type"] == "Counter" for c in ctrs), (
        "no Counter CTR descriptor in model.json — the config produced ZERO Counter "
        "splits and this fixture would pass trivially"
    )
    assert predictions.std() > 1e-6, "degenerate constant predictions"

    np.save(os.path.join(HERE, "X_cat.npy"), x_cat)
    np.save(os.path.join(HERE, "y.npy"), y.astype(np.float64))
    np.save(os.path.join(HERE, "predictions.npy"), predictions.astype(np.float64))

    config = {
        "scenario": SCENARIO,
        "requirement": "SPEC-CTRT-08",
        "seed": 0,
        "catboost_version": "1.2.10",
        "thread_count": 1,
        "n_rows": N_ROWS,
        "description": (
            "Counter-CTR end-to-end isolation (E12 / A3). Categorical-ONLY feature "
            "matrix (two columns, cardinality 6 and 5, zero float columns) so upstream "
            "float-quantization nondeterminism is structurally excluded. "
            "one_hot_max_size=1 routes both columns to the CTR path. "
            "simple_ctr=Counter with combinations_ctr=[] and max_ctr_complexity=1 "
            "isolates the simple Counter type: its numerator is the WHOLE-SET bucket "
            "total (each document's own row included) and its denominator is the "
            "constant MAX bucket total, so unlike every prefix type it is "
            "permutation-INdependent. NEVER regenerated in CI."
        ),
        "params": PARAMS,
        "cardinalities": {"cat0": CARD0, "cat1": CARD1},
        "stages": ["OnlineCtr", "Predict"],
        "npy_schema": {
            "X_cat.npy": "[N,2] int32 — categorical codes (stringified A4 form on the Rust side)",
            "y.npy": "[N] f64 — binclf label",
            "predictions.npy": "[N] f64 — RawFormulaVal (<=1e-5 gate)",
        },
        "note": (
            "FROZEN. Generated once by this directory's own gen_fixtures.py and "
            "NEVER regenerated in CI. Regenerating invalidates the <=1e-5 gate."
        ),
    }
    with open(os.path.join(HERE, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)
        fh.write("\n")

    # --- corpus-cleanliness guard --------------------------------------------
    # This generator must touch ONLY its own directory. A dirty path anywhere
    # else in the corpus means it reached outside its scenario.
    out = subprocess.run(
        ["git", "status", "--porcelain", FIXTURES],
        capture_output=True,
        text=True,
        cwd=FIXTURES,
    ).stdout
    offenders = [
        line
        for line in out.splitlines()
        if line.strip() and SCENARIO not in line
    ]
    if offenders:
        print("corpus contamination — this generator touched paths outside its scenario:")
        for line in offenders:
            print("   ", line)
        sys.exit(1)

    print(f"wrote {SCENARIO}: {N_ROWS} rows, predictions.std()={predictions.std():.6f}")
    print(f"ctr descriptors: {sorted({c['ctr_type'] for c in ctrs})}")


if __name__ == "__main__":
    main()
