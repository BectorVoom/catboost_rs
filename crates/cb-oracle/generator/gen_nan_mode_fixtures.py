"""Freeze the `nan_mode` parity fixture (Min / Max / Forbidden).

# What this pins

Upstream isolates missing values into their OWN quantization bin by adding a
SENTINEL border, and records the routing on the float feature:

    nan_mode=Min        -> f32::MIN PREPENDED to the borders, AsFalse
    nan_mode=Max        -> f32::MAX APPENDED  to the borders, AsTrue
    nan_mode=Forbidden  -> a NaN-bearing column is REJECTED at fit time
                           (quantization.cpp:320)

Before this wave the Rust fit path added NO sentinel at all, so NaN values
silently shared bin 0 with the smallest real values — a divergence on ANY
NaN-bearing dataset, not merely a missing parameter.

# Why the target is VALUE-driven, not missingness-driven

The obvious design — make `y` a function of WHETHER `f0` is missing — is
VACUOUS, and this generator's own guard caught it: `Min` and `Max` produced
byte-identical predictions. The reason is that the sentinel border isolates the
NaN bin COMPLETELY, so under either mode the tree simply learns "NaN -> c" and
the two modes agree on every row.

`Min` and `Max` can only diverge when NaN rows RIDE ALONG with real values at
feature 0's ORDINARY borders — that is, when the sentinel split is not what the
tree wants, so NaN falls to the bottom (`Min`) or the top (`Max`) of that
feature. So `y` here is driven by f0's VALUE (`3*f0 + 2*f1 - f2`), with NaN rows
behaving like a mid-range `f0`. Measured separation on the frozen eval matrix:
max |Min - Max| ~ 2.0 overall and ~1.99 on the NaN rows specifically.

# The eval set carries NaNs too

Predictions are frozen for a held-out matrix that includes NaN rows AND extreme
finite rows (+-1e30). Under `Min` a NaN must score with the LOW side and under
`Max` with the HIGH side, so the two modes' predictions must differ — which is
what makes the two frozen prediction vectors mutually discriminating.

Run:  python3 crates/cb-oracle/generator/gen_nan_mode_fixtures.py
"""

import json
import os

import numpy as np
from catboost import CatBoostRegressor

CATBOOST_VERSION = "1.2.10"

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "fixtures")
OUT_DIR = os.path.join(FIXTURES, "nan_mode")

SEED = 20260814
N_TRAIN = 300
N_FEATURES = 3
BORDER_COUNT = 16

# Pinned so the fit is deterministic and RNG-free (the isolating-params
# discipline used across this generator set).
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
    border_count=BORDER_COUNT,
    grow_policy="SymmetricTree",
)


def synthesize():
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(N_TRAIN, N_FEATURES)).astype(np.float64)
    # Feature 0 is the missing-bearing column (25% NaN).
    nan_mask = rng.random(N_TRAIN) < 0.25
    # VALUE-driven target (see the module doc): f0 drives y, and the NaN rows
    # behave like a mid-range f0, so the tree splits f0 at ORDINARY borders and
    # the NaN rows' side depends on the mode.
    f0 = np.where(nan_mask, 0.0, x[:, 0])
    y = (3.0 * f0 + 2.0 * x[:, 1] - x[:, 2]).astype(np.float64)
    x[nan_mask, 0] = np.nan
    return x, y, nan_mask


def eval_matrix():
    """Held-out rows: NaN rows, extreme-low rows, extreme-high rows, ordinary
    rows. The extremes are what separate Min from Max."""
    rng = np.random.default_rng(SEED + 1)
    ordinary = rng.normal(size=(12, N_FEATURES)).astype(np.float64)
    rows = [ordinary]
    nan_rows = rng.normal(size=(6, N_FEATURES)).astype(np.float64)
    nan_rows[:, 0] = np.nan
    rows.append(nan_rows)
    low = rng.normal(size=(3, N_FEATURES)).astype(np.float64)
    low[:, 0] = -1e30
    rows.append(low)
    high = rng.normal(size=(3, N_FEATURES)).astype(np.float64)
    high[:, 0] = 1e30
    rows.append(high)
    return np.ascontiguousarray(np.vstack(rows), dtype=np.float64)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    x, y, nan_mask = synthesize()
    x_eval = eval_matrix()

    np.save(os.path.join(OUT_DIR, "X.npy"), x)
    np.save(os.path.join(OUT_DIR, "y.npy"), y)
    np.save(os.path.join(OUT_DIR, "X_eval.npy"), x_eval)

    meta = {
        "scenario": "nan_mode",
        "catboost_version": CATBOOST_VERSION,
        "seed": SEED,
        "n_train": N_TRAIN,
        "n_features": N_FEATURES,
        "nan_feature_index": 0,
        "nan_fraction": float(nan_mask.mean()),
        "params": PARAMS,
        "target": "y = 3*f0 + 2*f1 - f2 with f0 imputed at 0 for the NaN rows -- "
                  "VALUE-driven on purpose, so Min and Max actually differ "
                  "(a missingness-driven target makes the two modes identical)",
        "eval_rows": "12 ordinary, 6 NaN-in-f0, 3 with f0=-1e30, 3 with f0=+1e30",
        "modes": {},
    }

    for nan_mode in ("Min", "Max"):
        model = CatBoostRegressor(nan_mode=nan_mode, **PARAMS)
        model.fit(x, y)
        preds = np.asarray(model.predict(x_eval), dtype=np.float64)
        np.save(os.path.join(OUT_DIR, "preds_%s.npy" % nan_mode), preds)

        model_path = os.path.join(OUT_DIR, "model_%s.json" % nan_mode)
        model.save_model(model_path, format="json")
        with open(model_path) as fh:
            doc = json.load(fh)
        ff = doc["features_info"]["float_features"]
        meta["modes"][nan_mode] = {
            "float_features": [
                {
                    "feature_index": f.get("feature_index"),
                    "has_nans": f.get("has_nans"),
                    "nan_value_treatment": f.get("nan_value_treatment"),
                    "n_borders": len(f.get("borders", [])),
                }
                for f in ff
            ],
        }
        print("%-4s preds[nan rows] = %s" % (nan_mode, np.round(preds[12:18], 6)))

    # Forbidden must be REJECTED, and the message is part of the contract.
    try:
        CatBoostRegressor(nan_mode="Forbidden", **PARAMS).fit(x, y)
        raise AssertionError("nan_mode=Forbidden must reject a NaN column")
    except AssertionError:
        raise
    except Exception as exc:
        meta["forbidden_rejection_message"] = " ".join(str(exc).split())
        print("Forbidden rejected:", meta["forbidden_rejection_message"][:120])

    min_p = np.load(os.path.join(OUT_DIR, "preds_Min.npy"))
    max_p = np.load(os.path.join(OUT_DIR, "preds_Max.npy"))
    meta["min_vs_max_max_abs_diff"] = float(np.max(np.abs(min_p - max_p)))
    if meta["min_vs_max_max_abs_diff"] == 0.0:
        raise AssertionError(
            "Min and Max produced IDENTICAL predictions: the fixture cannot tell "
            "the two modes apart and would pass for a wrong implementation"
        )
    print("Min vs Max max|diff| = %.6g (must be > 0 for the fixture to discriminate)"
          % meta["min_vs_max_max_abs_diff"])

    with open(os.path.join(OUT_DIR, "config.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    print("wrote", OUT_DIR)


if __name__ == "__main__":
    main()
