"""Generate the mixed simple-vs-combination CTR routing fixture (E17, SPEC-CTRT-10 / A5).

FROZEN GENERATOR — the committed `X_cat.npy` / `y.npy` / `model.json` /
`predictions.npy` / `config.json` under this directory are the GROUND TRUTH the
repo's `is_simple` CTR-type routing is compared against at <=1e-5. CI does NOT
run this script (no `catboost` install in CI) and never regenerates these
fixtures.

# Reproducibility caveat (load-bearing — do not "fix" by re-running this file)

CatBoost's quantization/border step has a documented source of run-to-run
nondeterminism independent of the exposed `random_seed` (observed even at
`thread_count=1`) — see `crates/cb-oracle/fixtures/ctr_load/gen_fixtures.py`.
This fixture sidesteps the FLOAT half of that problem structurally by using a
CATEGORICAL-ONLY feature matrix (no float columns => no float-border selection
at all), the same mitigation the other `ctr_*` fixtures use. The committed
artifacts ARE the fixtures; this script records the pinned recipe for
provenance.

# Why this fixture exists

`GetCtrInfo(projection)` routes a SINGLE-cat projection to `SimpleCtrs` and a
multi-cat projection to `TreeCtrs` (`ctr_helper.h:52-62`). This fixture pins
that discriminator end to end with DIFFERENT types AND DIFFERENT priors on the
two sides — a routing bug that lets one side's config govern the other cannot
agree with upstream on both the baked table types and the predictions.

# Pinned recipe

  - `catboost==1.2.10`, `numpy.random.RandomState(0)`.
  - 60 rows, THREE categorical columns (cardinality 6, 5, 4), ZERO float columns.
  - `one_hot_max_size=1`, so every column routes to the CTR path.
  - `simple_ctr=["Buckets:Prior=0.5"]`, `combinations_ctr=["Counter:Prior=0.25"]`,
    `max_ctr_complexity=2` — simple candidates are Buckets@0.5, combination
    candidates are Counter@0.25.
  - The target is CAT-DRIVEN with a genuine two-column interaction so upstream
    selects BOTH a simple and a combination CTR split; the guard asserts both.
  - The low-level `CatBoost(params)` API is used, NOT
    `CatBoostClassifier(**kwargs)`, so non-sklearn keys are honored.
"""

import json
import os
import subprocess
import sys

import numpy as np
from catboost import CatBoost, Pool

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.dirname(HERE)
SCENARIO = "ctr_mixed_simple_vs_combo"

N_ROWS = 60
CARDS = (6, 5, 4)

PARAMS = {
    "loss_function": "Logloss",
    "iterations": 10,
    "depth": 3,
    "learning_rate": 0.1,
    "l2_leaf_reg": 3.0,
    "boosting_type": "Plain",
    "one_hot_max_size": 1,
    "max_ctr_complexity": 2,
    "simple_ctr": ["Buckets:Prior=0.5"],
    "combinations_ctr": ["Counter:Prior=0.25"],
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

    c0 = rng.randint(0, CARDS[0], size=N_ROWS).astype(np.int32)
    c1 = rng.randint(0, CARDS[1], size=N_ROWS).astype(np.int32)
    c2 = rng.randint(0, CARDS[2], size=N_ROWS).astype(np.int32)
    x_cat = np.stack([c0, c1, c2], axis=1).astype(np.int32)

    # A random, NON-additive per-(cat0, cat1)-combination effect table (no
    # decomposition into independent per-feature terms) so the label genuinely
    # needs the combined projection, not two separate simple CTRs — the same
    # design that makes the committed `fstr_ctr` fixture select a combination
    # CTR. Labels are NOISE-FREE (`logit > 0` after centering): additive label
    # noise dilutes the interaction below the greedy scorer's selection
    # threshold at this scale (verified by a local param sweep — the
    # rand()<sigmoid variant never selects a combination split).
    tbl = rng.normal(0.0, 2.5, size=(CARDS[0], CARDS[1]))
    logit = 0.4 * (c0 % 2) - 0.3 * (c1 % 3) + tbl[c0, c1] + 0.2 * (c2 % 2)
    logit = logit - logit.mean()
    y = (logit > 0).astype(np.int32)

    if len(np.unique(y)) < 2:
        sys.exit("degenerate target: only one class present")

    pool = Pool(x_cat, label=y, cat_features=[0, 1, 2])
    model = CatBoost(PARAMS)
    model.fit(pool)

    predictions = model.predict(pool, prediction_type="RawFormulaVal")

    model_json_path = os.path.join(HERE, "model.json")
    model.save_model(model_json_path, format="json")
    with open(model_json_path) as fh:
        model_json = json.load(fh)

    # --- MANDATORY anti-false-pass guard --------------------------------------
    # The routing is only testable if the committed model carries BOTH kinds,
    # each with its own configured type. If this fires, strengthen the
    # interaction signal — do NOT weaken the assertion.
    ctrs = model_json["features_info"].get("ctrs", [])
    simple = [c for c in ctrs if len(c["elements"]) == 1]
    combo = [c for c in ctrs if len(c["elements"]) >= 2]
    assert simple and combo, (
        f"mixed fixture needs BOTH a simple and a combination CTR; got "
        f"{len(simple)} simple / {len(combo)} combination"
    )
    assert {c["ctr_type"] for c in simple} == {"Buckets"}, (
        f"simple CTRs must be Buckets, got { {c['ctr_type'] for c in simple} }"
    )
    assert {c["ctr_type"] for c in combo} == {"Counter"}, (
        f"combination CTRs must be Counter, got { {c['ctr_type'] for c in combo} }"
    )
    assert predictions.std() > 1e-6, "degenerate constant predictions"

    np.save(os.path.join(HERE, "X_cat.npy"), x_cat)
    np.save(os.path.join(HERE, "y.npy"), y.astype(np.float64))
    np.save(os.path.join(HERE, "predictions.npy"), predictions.astype(np.float64))

    config = {
        "scenario": SCENARIO,
        "requirement": "SPEC-CTRT-10",
        "seed": 0,
        "catboost_version": "1.2.10",
        "thread_count": 1,
        "n_rows": N_ROWS,
        "description": (
            "Mixed simple-vs-combination CTR routing (E17 / A5). Categorical-ONLY "
            "feature matrix (three columns, cardinality 6/5/4, zero float columns). "
            "simple_ctr=Buckets:Prior=0.5 vs combinations_ctr=Counter:Prior=0.25 at "
            "max_ctr_complexity=2 pin the is_simple discriminator end to end: types "
            "AND priors differ across the two sides, so cross-routing cannot agree "
            "with upstream. NEVER regenerated in CI."
        ),
        "params": PARAMS,
        "cardinalities": {"cat0": CARDS[0], "cat1": CARDS[1], "cat2": CARDS[2]},
        "stages": ["OnlineCtr", "CandidateExpansion", "Predict"],
        "observed": {
            "simple_ctr_types": sorted({c["ctr_type"] for c in simple}),
            "combo_ctr_types": sorted({c["ctr_type"] for c in combo}),
            "n_simple": len(simple),
            "n_combo": len(combo),
        },
        "npy_schema": {
            "X_cat.npy": "[N,3] int32 — categorical codes (stringified A4 form on the Rust side)",
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

    print(f"wrote {SCENARIO}: {N_ROWS} rows, predictions.std()={predictions.std():.6f}")
    print(f"simple: {sorted({c['ctr_type'] for c in simple})} x{len(simple)}; "
          f"combo: {sorted({c['ctr_type'] for c in combo})} x{len(combo)}")


if __name__ == "__main__":
    main()
