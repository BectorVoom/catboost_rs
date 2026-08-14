"""Freeze a per-VALUE oracle for every string-valued parameter that was ALREADY
implemented before this wave.

Each new parameter in the wave got its own dedicated fixture. This one closes
the other half of the requirement -- "oracle tests for all string-valued
parameters" -- by pinning one prediction vector per (parameter, value) cell for
the params that pre-date it:

    loss_function          RMSE, MAE, LogCosh          (regression corpus)
                           Logloss, CrossEntropy       (binary corpus)
    score_function         Cosine, L2, SolarL2, NewtonL2, NewtonCosine,
                           LOOL2, SatL2
    leaf_estimation_method Gradient, Newton, Simple
    grow_policy            SymmetricTree, Depthwise, Lossguide, Region
    bootstrap_type         No, Bayesian, Bernoulli, MVS, Poisson
    boosting_type          Plain, Ordered

Only values this crate's Python binding actually accepts are enumerated -- the
loss list is exactly `parse_loss`'s, so the matrix cannot drift into pinning
losses the binding rejects.

Cells that catboost itself REFUSES (illegal parameter combinations) are recorded
as skips WITH the upstream reason rather than silently dropped, so the matrix
documents what is genuinely unavailable instead of quietly shrinking.

Run:  python3 crates/cb-oracle/generator/gen_string_param_matrix.py
"""

import json
import os

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor

CATBOOST_VERSION = "1.2.10"

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "fixtures")
OUT_DIR = os.path.join(FIXTURES, "string_param_matrix")

SEED = 20260814
N_TRAIN = 300
N_FEATURES = 4

# Every confound pinned; each cell overrides exactly ONE parameter.
BASE = dict(
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
    border_count=32,
    grow_policy="SymmetricTree",
    boosting_type="Plain",
)

# The sampler knobs are PER BOOTSTRAP TYPE and must be pinned explicitly, but
# cannot be pinned globally:
#
#   * catboost's defaults differ from this crate's builder defaults
#     (bagging_temperature 1.0 vs 0.0, subsample 0.8 vs 1.0) and BOTH crate
#     defaults are NO-OPS -- leaving them unset made every bootstrap cell
#     reproduce the `No` baseline on the Rust side while catboost sampled;
#   * but upstream REJECTS `subsample` under `No`/`Bayesian`, so a global pin
#     makes those cells un-generatable.
#
# So each bootstrap type carries exactly the knobs it accepts. Same class of trap
# as the cv/ORCH-01 fixture: pin every default the two sides disagree on, on BOTH
# sides.
BOOTSTRAP_KNOBS = {
    "No": {},
    "Bayesian": {"bagging_temperature": 1.0},
    "Bernoulli": {"subsample": 0.8},
    "MVS": {"subsample": 0.8},
    "Poisson": {"subsample": 0.8},
}

# (param, value, corpus) -- corpus is "reg" or "bin".
CELLS = []
for v in ("RMSE", "MAE", "LogCosh"):
    CELLS.append(("loss_function", v, "reg"))
for v in ("Logloss", "CrossEntropy"):
    CELLS.append(("loss_function", v, "bin"))
for v in ("Cosine", "L2", "SolarL2", "NewtonL2", "NewtonCosine", "LOOL2", "SatL2"):
    CELLS.append(("score_function", v, "reg"))
# leaf_estimation_method needs a loss whose SECOND derivative is not constant:
# under RMSE the Newton step equals the Gradient step (and Simple is Gradient by
# definition), so all three cells came out identical and the parameter was not
# discriminated at all. Logloss separates them.
for v in ("Gradient", "Newton", "Simple"):
    CELLS.append(("leaf_estimation_method", v, "bin"))
for v in ("SymmetricTree", "Depthwise", "Lossguide", "Region"):
    CELLS.append(("grow_policy", v, "reg"))
for v in ("No", "Bayesian", "Bernoulli", "MVS", "Poisson"):
    CELLS.append(("bootstrap_type", v, "reg"))
for v in ("Plain", "Ordered"):
    CELLS.append(("boosting_type", v, "reg"))


def corpora():
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(N_TRAIN, N_FEATURES)).astype(np.float64)
    lin = 3.0 * x[:, 0] + 2.0 * x[:, 1] - x[:, 2] + 0.5 * x[:, 3]
    y_reg = lin.astype(np.float64)
    y_bin = (lin > 0).astype(np.float64)
    x_eval = np.ascontiguousarray(rng.normal(size=(12, N_FEATURES)), dtype=np.float64)
    return x, y_reg, y_bin, x_eval


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    x, y_reg, y_bin, x_eval = corpora()
    np.save(os.path.join(OUT_DIR, "X.npy"), x)
    np.save(os.path.join(OUT_DIR, "y_reg.npy"), y_reg)
    np.save(os.path.join(OUT_DIR, "y_bin.npy"), y_bin)
    np.save(os.path.join(OUT_DIR, "X_eval.npy"), x_eval)

    meta = {
        "scenario": "string_param_matrix",
        "catboost_version": CATBOOST_VERSION,
        "seed": SEED,
        "base_params": BASE,
        "prediction_type": "RawFormulaVal",
        "cells": {},
        "skipped": {},
    }

    written = 0
    for param, value, corpus in CELLS:
        stem = "%s__%s" % (param, value)
        over = dict(BASE)
        over[param] = value
        if param == "bootstrap_type":
            over.update(BOOTSTRAP_KNOBS.get(value, {}))
        if corpus == "bin":
            cls, y = CatBoostClassifier, y_bin
            # boost_from_average is not on upstream's allow-list for every loss;
            # keep it off for the binary cells so the cell is about the param.
            over["boost_from_average"] = False
        else:
            cls, y = CatBoostRegressor, y_reg
            over.setdefault("loss_function", "RMSE")
            over["boost_from_average"] = False
        try:
            m = cls(**over)
            m.fit(x, y)
            p = np.asarray(
                m.predict(x_eval, prediction_type="RawFormulaVal"), dtype=np.float64
            )
        except Exception as exc:
            meta["skipped"][stem] = " ".join(str(exc).split())[:220]
            print("SKIP %-42s %s" % (stem, meta["skipped"][stem][:80]))
            continue
        if not np.all(np.isfinite(p)):
            meta["skipped"][stem] = "catboost produced non-finite predictions"
            print("SKIP %-42s non-finite" % stem)
            continue
        np.save(os.path.join(OUT_DIR, "preds_%s.npy" % stem), p)
        meta["cells"][stem] = {
            "param": param,
            "value": value,
            "corpus": corpus,
            "loss_function": over.get("loss_function", "RMSE"),
            "bagging_temperature": over.get("bagging_temperature"),
            "subsample": over.get("subsample"),
        }
        written += 1
        print("ok   %-42s preds[:2]=%s" % (stem, np.round(p[:2], 6)))

    # Every parameter must retain at least TWO distinct cells, otherwise the
    # matrix cannot tell that parameter's values apart at all.
    per_param = {}
    for cell in meta["cells"].values():
        per_param.setdefault(cell["param"], []).append(cell["value"])
    for param, values in sorted(per_param.items()):
        if len(values) < 2:
            raise AssertionError(
                "parameter %s kept only %d cell(s) (%s); it cannot be discriminated"
                % (param, len(values), values)
            )
    meta["values_per_param"] = {k: sorted(v) for k, v in per_param.items()}

    with open(os.path.join(OUT_DIR, "config.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    print("\nwrote %d cells (%d skipped) to %s"
          % (written, len(meta["skipped"]), OUT_DIR))


if __name__ == "__main__":
    main()
