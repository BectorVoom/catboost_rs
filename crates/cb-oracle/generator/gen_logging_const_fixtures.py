"""Freeze the logging-family + `allow_const_label` parity facts.

# The logging family is NUMERICALLY INERT

`logging_level` (Silent / Verbose / Info / Debug), `verbose`, `silent` and
`metric_period` are OUTPUT controls. Measured against catboost 1.2.10, every one
of them leaves the trained model byte-identical (max |diff| = 0). That is the
property worth pinning: it is what licenses accepting them without emitting any
log, and it is what would break if one of them were ever wired into training.

Upstream also enforces a cross-parameter rule:

    Only one of parameters ['verbose', 'logging_level', 'verbose_eval',
    'silent'] should be set

`metric_period` is NOT a member -- it may be combined with `verbose`.

# allow_const_label

A learn set whose targets are all equal is REFUSED by default
(`metric.cpp:7011`, "All train targets are equal"); with the flag it trains and
predicts that constant.

Run:  python3 crates/cb-oracle/generator/gen_logging_const_fixtures.py
"""

import json
import os

import numpy as np
from catboost import CatBoostRegressor

CATBOOST_VERSION = "1.2.10"

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "fixtures")
OUT_DIR = os.path.join(FIXTURES, "logging_const_label")

SEED = 20260814
N_TRAIN = 200
N_FEATURES = 3
CONST_TARGET = 3.5

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
    border_count=32,
    boost_from_average=True,
)

# Each is applied ALONE (the mutual-exclusion rule forbids combining the first
# four); `metric_period` is paired with `verbose` because upstream allows that.
LOGGING_SETTINGS = [
    {"verbose": False},
    {"logging_level": "Silent"},
    {"logging_level": "Verbose"},
    {"logging_level": "Info"},
    {"logging_level": "Debug"},
    {"silent": True},
    {"verbose": True},
    {"verbose": False, "metric_period": 3},
]

EXCLUSIVE_COMBOS = [
    {"verbose": False, "logging_level": "Silent"},
    {"verbose": False, "silent": True},
    {"silent": True, "logging_level": "Silent"},
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(N_TRAIN, N_FEATURES)).astype(np.float64)
    y = (2.0 * x[:, 0] - x[:, 1]).astype(np.float64)
    x_eval = np.ascontiguousarray(rng.normal(size=(8, N_FEATURES)), dtype=np.float64)

    np.save(os.path.join(OUT_DIR, "X.npy"), x)
    np.save(os.path.join(OUT_DIR, "y.npy"), y)
    np.save(os.path.join(OUT_DIR, "X_eval.npy"), x_eval)

    meta = {
        "scenario": "logging_const_label",
        "catboost_version": CATBOOST_VERSION,
        "seed": SEED,
        "params": PARAMS,
        "const_target": CONST_TARGET,
        "logging_levels": ["Silent", "Verbose", "Info", "Debug"],
        "mutually_exclusive": ["verbose", "logging_level", "verbose_eval", "silent"],
    }

    # Baseline + every logging setting must agree EXACTLY.
    base = CatBoostRegressor(verbose=False, **PARAMS)
    base.fit(x, y)
    base_p = np.asarray(base.predict(x_eval), dtype=np.float64)
    np.save(os.path.join(OUT_DIR, "preds.npy"), base_p)

    worst = 0.0
    for setting in LOGGING_SETTINGS:
        m = CatBoostRegressor(**dict(PARAMS, **setting))
        m.fit(x, y)
        p = np.asarray(m.predict(x_eval), dtype=np.float64)
        d = float(np.max(np.abs(p - base_p)))
        worst = max(worst, d)
        if d != 0.0:
            raise AssertionError(
                "logging setting %s changed the model by %g; it must be inert"
                % (setting, d)
            )
    meta["logging_max_abs_diff_over_all_settings"] = worst
    print("all %d logging settings are numerically inert (worst |diff| = %g)"
          % (len(LOGGING_SETTINGS), worst))

    # The mutual-exclusion rule.
    rejections = {}
    for combo in EXCLUSIVE_COMBOS:
        try:
            CatBoostRegressor(**dict(PARAMS, **combo)).fit(x, y)
            raise AssertionError("combo %s must be rejected" % combo)
        except AssertionError:
            raise
        except Exception as exc:
            rejections[json.dumps(combo, sort_keys=True)] = " ".join(str(exc).split())
    meta["mutual_exclusion_rejections"] = rejections
    print("mutual exclusion enforced for %d combos" % len(rejections))
    # verbose + metric_period is allowed.
    CatBoostRegressor(**dict(PARAMS, verbose=1, metric_period=2)).fit(x, y)
    print("verbose + metric_period accepted (metric_period is NOT in the rule)")

    # allow_const_label.
    const = np.full(N_TRAIN, CONST_TARGET, dtype=np.float64)
    np.save(os.path.join(OUT_DIR, "y_const.npy"), const)
    try:
        CatBoostRegressor(verbose=False, **PARAMS).fit(x, const)
        raise AssertionError("a constant target must be refused by default")
    except AssertionError:
        raise
    except Exception as exc:
        meta["const_label_rejection"] = " ".join(str(exc).split())
        print("constant target refused:", meta["const_label_rejection"][:100])

    allowed = CatBoostRegressor(allow_const_label=True, verbose=False, **PARAMS)
    allowed.fit(x, const)
    cp = np.asarray(allowed.predict(x_eval), dtype=np.float64)
    np.save(os.path.join(OUT_DIR, "preds_const_allowed.npy"), cp)
    meta["const_allowed_prediction"] = float(cp[0])
    meta["const_allowed_tree_count"] = int(allowed.tree_count_)
    print("allow_const_label=True trains %d trees predicting %.6f"
          % (allowed.tree_count_, cp[0]))

    with open(os.path.join(OUT_DIR, "config.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    print("wrote", OUT_DIR)


if __name__ == "__main__":
    main()
