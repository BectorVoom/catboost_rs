"""Generate the multi-prior Borders-CTR parity fixture (E14, SPEC-CTRT-11 / A4).

FROZEN GENERATOR — the committed `X_cat.npy` / `y.npy` / `model.json` /
`predictions.npy` / `config.json` under this directory are the GROUND TRUTH the
repo's CTR candidate-expansion path is compared against at <=1e-5. CI does NOT
run this script (no `catboost` install in CI) and never regenerates these
fixtures.

# Reproducibility caveat (load-bearing — do not "fix" by re-running this file)

CatBoost's quantization/border step has a documented source of run-to-run
nondeterminism independent of the exposed `random_seed` (observed even at
`thread_count=1`) — see `crates/cb-oracle/fixtures/ctr_load/gen_fixtures.py`.
This fixture sidesteps the FLOAT half of that problem structurally by using a
CATEGORICAL-ONLY feature matrix (no float columns => no float-border selection
at all), the same mitigation `tensor_ctr_e2e` and `ctr_btmv_simple` use. The
committed artifacts ARE the fixtures; this script records the pinned recipe for
provenance.

# Why this fixture exists

The engine emitted exactly ONE candidate column per projection, built from
`priors.first()` — the whole configured prior LIST beyond its first element was
inert. Upstream emits one candidate per `(ctrIdx, targetBorderIdx, priorIdx)`
(`greedy_tensor_search.cpp:414-427`), so a multi-prior configuration scores a
strictly larger candidate set and can pick a different split. This fixture pins
that expansion end to end: a model whose splits genuinely live at more than one
prior cannot be reproduced by a single-prior engine.

# Pinned recipe

  - `catboost==1.2.10`, `numpy.random.RandomState(0)`.
  - 120 rows, TWO categorical columns (cardinality 8 and 6), ZERO float columns.
  - `one_hot_max_size=1`, so both columns route to the CTR path rather than
    one-hot (`route_categorical(card, 1) == Ctr`).
  - `simple_ctr=["Borders:Prior=0:Prior=0.5:Prior=1"]`, `combinations_ctr=[]`,
    `max_ctr_complexity=1` — simple Borders CTRs at THREE priors.
  - The target is CAT-DRIVEN so upstream actually selects CTR splits at more
    than one prior; a weak categorical signal collapses onto a single prior and
    the fixture would pass trivially. The anti-false-pass guard below asserts
    that outcome away.
  - The low-level `CatBoost(params)` API is used, NOT
    `CatBoostClassifier(**kwargs)`, so non-sklearn keys such as `simple_ctr`
    are honored.

# Note on the model.json key names

Upstream writes the prior as `prior_numerator` / `prior_denomerator` (sic) in
`features_info.ctrs`, not `prior_num` / `prior_denom`. The guard below reads the
real keys.
"""

import json
import os
import subprocess
import sys

import numpy as np
from catboost import CatBoost, Pool

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.dirname(HERE)
SCENARIO = "ctr_borders_multiprior"

N_ROWS = 120
CARD0 = 8
CARD1 = 6

PARAMS = {
    "loss_function": "Logloss",
    "iterations": 20,
    "depth": 3,
    "learning_rate": 0.1,
    "l2_leaf_reg": 3.0,
    "boosting_type": "Plain",
    "one_hot_max_size": 1,
    "max_ctr_complexity": 1,
    "simple_ctr": ["Borders:Prior=0:Prior=0.5:Prior=1"],
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
    # upstream genuinely selects CTR splits — at several priors.
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
    # Multi-prior expansion is only testable if upstream genuinely selected
    # splits at more than one prior. If this fires, widen the corpus (more rows /
    # stronger cat signal) — do NOT weaken the assertion.
    ctrs = model_json["features_info"].get("ctrs", [])
    borders = [c for c in ctrs if c["ctr_type"] == "Borders"]
    assert borders, "no Borders CTR descriptor in model.json — fixture is vacuous"
    priors = sorted(
        {round(c["prior_numerator"] / c["prior_denomerator"], 6) for c in borders}
    )
    assert len(priors) >= 2, (
        f"multi-prior expansion is untestable: model.json carries priors {priors}; "
        "the config produced fewer than two distinct prior columns"
    )
    assert predictions.std() > 1e-6, "degenerate constant predictions"

    np.save(os.path.join(HERE, "X_cat.npy"), x_cat)
    np.save(os.path.join(HERE, "y.npy"), y.astype(np.float64))
    np.save(os.path.join(HERE, "predictions.npy"), predictions.astype(np.float64))

    config = {
        "scenario": SCENARIO,
        "requirement": "SPEC-CTRT-11",
        "seed": 0,
        "catboost_version": "1.2.10",
        "thread_count": 1,
        "n_rows": N_ROWS,
        "description": (
            "Multi-prior Borders-CTR candidate expansion (E14/E15 / A4). Categorical-ONLY "
            "feature matrix (two columns, cardinality 8 and 6, zero float columns) so "
            "upstream float-quantization nondeterminism is structurally excluded. "
            "one_hot_max_size=1 routes both columns to the CTR path. "
            "simple_ctr=Borders at THREE priors (0, 0.5, 1) with combinations_ctr=[] and "
            "max_ctr_complexity=1 isolates the (projection, prior) candidate product: "
            "upstream emits one candidate column per prior, so a single-prior engine "
            "scores a strictly smaller candidate set and picks different splits. "
            "NEVER regenerated in CI."
        ),
        "params": PARAMS,
        "cardinalities": {"cat0": CARD0, "cat1": CARD1},
        "stages": ["OnlineCtr", "CandidateExpansion", "Predict"],
        "observed_priors": priors,
        "double_generation_check": (
            "Two consecutive runs of this generator produced byte-identical "
            "X_cat.npy / y.npy / predictions.npy and a structurally identical "
            "model.json. The ONLY model.json divergence is upstream's volatile "
            "metadata — `model_guid` and `train_finish_time` — which no oracle "
            "reads. The reference itself is therefore deterministic."
        ),
        "npy_schema": {
            "X_cat.npy": "[N,2] int32 — categorical codes (stringified A4 form on the Rust side)",
            "y.npy": "[N] f64 — binclf label",
            "predictions.npy": "[N] f64 — RawFormulaVal (<=1e-5 gate)",
        },
        "note": (
            "FROZEN. Generated once by this directory's own gen_fixtures.py and "
            "NEVER regenerated in CI. Regenerating invalidates the <=1e-5 gate. "
            f"The anti-false-pass guard observed distinct priors {priors} across "
            "the committed model.json's Borders descriptors."
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
        line for line in out.splitlines() if line.strip() and SCENARIO not in line
    ]
    if offenders:
        print("corpus contamination — this generator touched paths outside its scenario:")
        for line in offenders:
            print("   ", line)
        sys.exit(1)

    print(f"wrote {SCENARIO}: {N_ROWS} rows, predictions.std()={predictions.std():.6f}")
    print(f"ctr descriptors: {len(ctrs)}; distinct Borders priors: {priors}")


if __name__ == "__main__":
    main()
