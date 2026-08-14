"""Freeze the `sampling_frequency` / `sampling_unit` parity facts.

Two facts are pinned, and they are what license this engine's accept/refuse split.

# 1. `sampling_frequency` is INERT without a draw

Under `bootstrap_type="No"` the sampler never draws, so there is no sample to
redraw and `PerTree` / `PerTreeLevel` must produce byte-identical models. Verified
at several depths -- this is why the engine ACCEPTS `PerTreeLevel` in that regime
instead of refusing an inert value.

# 2. `PerTree` (what this engine implements) matches upstream under a DRAWING sampler

The `preds_*` arrays are the parity target for the engine's `PerTree` behaviour with
Bernoulli / Bayesian / MVS sampling. Without these the "we implement PerTree" claim
would rest on the default never being exercised.

# 3. `PerTreeLevel` genuinely DIFFERS under a drawing sampler -- including at depth 1

Recorded as evidence for the refusal. The depth-1 cell is the important one: there
both frequencies make exactly ONE draw, so a non-zero diff proves the divergence is
the draw's POSITION in the RNG stream, not its count -- which is precisely why
"move the bootstrap call into the level loop" would NOT reproduce upstream.

# Vacuity guards

The generator REFUSES to write a fixture where the facts it claims are not actually
exhibited: the inert cells must be exactly equal, and the divergent cells must
actually diverge.

Run:  python3 crates/cb-oracle/generator/gen_sampling_fixtures.py
"""

import json
import os

import numpy as np
from catboost import CatBoostRegressor

CATBOOST_VERSION = "1.2.10"

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "fixtures")
OUT_DIR = os.path.join(FIXTURES, "sampling")

SEED = 20260815
N_TRAIN = 400
N_FEATURES = 4
N_EVAL = 10

BASE = dict(
    iterations=5,
    depth=3,
    learning_rate=0.3,
    l2_leaf_reg=3.0,
    random_strength=0,
    leaf_estimation_iterations=1,
    score_function="L2",
    leaf_estimation_method="Gradient",
    random_seed=0,
    thread_count=1,
    border_count=32,
    boost_from_average=False,
    verbose=False,
)

#: Each drawing sampler carries EXACTLY the knobs upstream accepts for it -- a global
#: pin is impossible because `No`/`Bayesian` reject `subsample`. Same trap as the
#: cv/ORCH-01 fixture: pin every default the two sides disagree on, on BOTH sides.
DRAWING_SAMPLERS = {
    "Bernoulli": {"subsample": 0.7},
    "Bayesian": {"bagging_temperature": 1.0},
    "MVS": {"subsample": 0.7},
}

INERT_DEPTHS = (1, 2, 4)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(N_TRAIN, N_FEATURES)).astype(np.float64)
    y = (2.0 * x[:, 0] - x[:, 1] + 0.5 * x[:, 2]).astype(np.float64)
    x_eval = np.ascontiguousarray(rng.normal(size=(N_EVAL, N_FEATURES)), dtype=np.float64)

    np.save(os.path.join(OUT_DIR, "X.npy"), x)
    np.save(os.path.join(OUT_DIR, "y.npy"), y)
    np.save(os.path.join(OUT_DIR, "X_eval.npy"), x_eval)

    def fit(**over):
        m = CatBoostRegressor(**dict(BASE, **over))
        m.fit(x, y)
        return np.asarray(m.predict(x_eval, prediction_type="RawFormulaVal"), dtype=np.float64)

    meta = {
        "scenario": "sampling",
        "catboost_version": CATBOOST_VERSION,
        "seed": SEED,
        "params": BASE,
        "drawing_samplers": DRAWING_SAMPLERS,
        "sampling_units": ["Object", "Group"],
        "sampling_frequencies": ["PerTree", "PerTreeLevel"],
    }

    # ---- Fact 1: inert without a draw -------------------------------------------
    inert = {}
    for depth in INERT_DEPTHS:
        a = fit(depth=depth, bootstrap_type="No", sampling_frequency="PerTree")
        b = fit(depth=depth, bootstrap_type="No", sampling_frequency="PerTreeLevel")
        d = float(np.max(np.abs(a - b)))
        if d != 0.0:
            raise AssertionError(
                "bootstrap_type=No depth=%d: PerTree vs PerTreeLevel differ by %g; "
                "the no-draw carve-out this fixture licenses would be WRONG" % (depth, d)
            )
        inert["depth_%d" % depth] = d
    meta["inert_without_draw_max_abs_diff"] = inert
    print("fact 1 OK: sampling_frequency inert under bootstrap_type=No at depths %s"
          % (INERT_DEPTHS,))

    # ---- Fact 2: PerTree parity target under each drawing sampler ---------------
    for name, knobs in DRAWING_SAMPLERS.items():
        p = fit(bootstrap_type=name, sampling_frequency="PerTree", **knobs)
        np.save(os.path.join(OUT_DIR, "preds_pertree_%s.npy" % name.lower()), p)
    print("fact 2 OK: wrote PerTree parity targets for %s" % ", ".join(DRAWING_SAMPLERS))

    # ---- Fact 3: PerTreeLevel really differs, including at depth 1 --------------
    divergence = {}
    for name, knobs in DRAWING_SAMPLERS.items():
        for depth in (1, 3):
            a = fit(depth=depth, bootstrap_type=name, sampling_frequency="PerTree", **knobs)
            b = fit(depth=depth, bootstrap_type=name, sampling_frequency="PerTreeLevel", **knobs)
            d = float(np.max(np.abs(a - b)))
            if d == 0.0:
                raise AssertionError(
                    "%s depth=%d: PerTree and PerTreeLevel coincide, so this cell is "
                    "VACUOUS as evidence for the refusal" % (name, depth)
                )
            divergence["%s_depth_%d" % (name.lower(), depth)] = d
    meta["pertreelevel_divergence_max_abs_diff"] = divergence
    print("fact 3 OK: PerTreeLevel diverges under every drawing sampler, depth 1 included")
    print("    depth-1 diffs prove the divergence is RNG POSITION, not draw count:")
    for k, v in divergence.items():
        if k.endswith("_depth_1"):
            print("      %-22s %.6g" % (k, v))

    # ---- sampling_unit=Group is refused by upstream on an ungrouped pool --------
    try:
        fit(bootstrap_type="Bernoulli", subsample=0.7, sampling_unit="Group")
        raise AssertionError("sampling_unit=Group must be refused on an ungrouped pool")
    except AssertionError:
        raise
    except Exception as exc:
        meta["group_sampling_unit_rejection"] = " ".join(str(exc).split())
        print("sampling_unit=Group refused upstream: %s"
              % meta["group_sampling_unit_rejection"][:110])

    with open(os.path.join(OUT_DIR, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    print("\nwrote %s" % OUT_DIR)


if __name__ == "__main__":
    main()
