"""Freeze the `rsm` (`colsample_bylevel`) parity facts against catboost 1.2.10.

`rsm` offers only a FRACTION of the features to the split search at each tree
LEVEL. Upstream draws one `GenRandReal1()` per listed candidate from the
PERSISTENT learn RNG and keeps the candidate iff `draw <= rsm`
(`SelectCandidatesAndCleanupStatsFromPrevTree`, `greedy_tensor_search.cpp:334`),
so `rsm` is not merely a feature filter -- it shares the RNG stream with
`random_strength` and the bootstrap, and getting the draw COUNT or ORDER wrong
desynchronises everything downstream.

Facts pinned here:

# 1. `rsm = 1.0` is INERT

Byte-identical to leaving `rsm` unset, INCLUDING under a drawing bootstrap. This
is what licenses the engine's `rsm_active = rsm < 1.0` gate: at the default the
zero-draw fast path must stay untouched.

# 2. The parity targets

`preds_rsm_*` are the predictions to match for each subsampling fraction, at
`bootstrap_type=No` / `random_strength=0` so the ONLY RNG consumer is `rsm`
itself. That isolation is the point: a draw-accounting bug cannot hide behind
bootstrap noise.

# 3. Trees really do END EARLY when a level selects nothing

At a small enough `rsm` a level's candidate list comes back empty and upstream
`break`s the depth loop (`greedy_tensor_search.cpp:1209`), so a depth-`d` request
yields a tree with FEWER than `d` splits. `tree_split_counts` records the actual
per-tree split counts; without this the engine's early-stop rule would be
unverified guesswork.

# 4. `rsm` composes with the other RNG consumers

Cells with a drawing bootstrap and with `random_strength != 0` pin that the
draws interleave in the right ORDER, not just the right count.

# Vacuity guards

The generator REFUSES to write a fixture whose claims are not exhibited: the
inert cells must be exactly equal, every distinct `rsm` must produce a distinct
model, and the early-stop cell must actually contain a short tree.

Run:  python3 crates/cb-oracle/generator/gen_rsm_fixtures.py
"""

import json
import os

import numpy as np
from catboost import CatBoostRegressor

CATBOOST_VERSION = "1.2.10"

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "fixtures")
OUT_DIR = os.path.join(FIXTURES, "rsm")

SEED = 20260815
N_TRAIN = 400
N_FEATURES = 6
N_EVAL = 12

#: Pinned EXPLICITLY on both sides -- every default the raw dict API and the
#: builder could disagree on (the `random_strength=0` trap from the cv/ORCH-01
#: fixture: catboost's own default is NOT 0).
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
    bootstrap_type="No",
    grow_policy="SymmetricTree",
    verbose=False,
    allow_writing_files=False,
)

#: The parity targets. Every one must yield a DISTINCT model (guard below).
RSM_VALUES = (1.0, 0.75, 0.5, 0.25)

#: `rsm` low enough that some level selects nothing and the tree ends early.
EARLY_STOP_RSM = 0.1

#: Cells proving the draws interleave correctly with the OTHER RNG consumers.
COMPOSITION_CELLS = {
    "bernoulli": dict(bootstrap_type="Bernoulli", subsample=0.7),
    "bayesian": dict(bootstrap_type="Bayesian", bagging_temperature=1.0),
    "random_strength": dict(random_strength=1.0),
}


def tree_split_counts(model):
    """Per-tree split count, read back from the saved JSON model."""
    path = os.path.join(OUT_DIR, "_tmp_model.json")
    model.save_model(path, format="json")
    with open(path) as fh:
        mj = json.load(fh)
    counts = [len(t.get("splits") or []) for t in mj.get("oblivious_trees", [])]
    os.remove(path)
    return counts


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(N_TRAIN, N_FEATURES)).astype(np.float64)
    # Every feature carries signal, so dropping one genuinely changes the split
    # choice -- a target depending on one column only would make `rsm` look inert
    # whenever that column survives (the fixture-vacuity trap this wave keeps hitting).
    y = (
        2.0 * x[:, 0] - 1.3 * x[:, 1] + 0.9 * x[:, 2]
        - 0.7 * x[:, 3] + 0.5 * x[:, 4] - 0.4 * x[:, 5]
    ).astype(np.float64)
    x_eval = np.ascontiguousarray(rng.normal(size=(N_EVAL, N_FEATURES)), dtype=np.float64)

    np.save(os.path.join(OUT_DIR, "X.npy"), x)
    np.save(os.path.join(OUT_DIR, "y.npy"), y)
    np.save(os.path.join(OUT_DIR, "X_eval.npy"), x_eval)

    def fit(**over):
        m = CatBoostRegressor(**dict(BASE, **over))
        m.fit(x, y)
        return m

    def preds(**over):
        m = fit(**over)
        return np.asarray(
            m.predict(x_eval, prediction_type="RawFormulaVal"), dtype=np.float64
        )

    meta = {
        "scenario": "rsm",
        "catboost_version": CATBOOST_VERSION,
        "seed": SEED,
        "n_features": N_FEATURES,
        "params": BASE,
        "rsm_values": list(RSM_VALUES),
        "early_stop_rsm": EARLY_STOP_RSM,
    }

    # ---- Fact 1: rsm = 1.0 is inert ------------------------------------------
    inert = {}
    unset = preds()
    one = preds(rsm=1.0)
    d = float(np.max(np.abs(unset - one)))
    if d != 0.0:
        raise AssertionError(
            "rsm=1.0 differs from unset by %g; the engine's `rsm_active = rsm < 1.0` "
            "gate would be WRONG" % d
        )
    inert["bootstrap_no"] = d
    for name, knobs in COMPOSITION_CELLS.items():
        a = preds(**knobs)
        b = preds(rsm=1.0, **knobs)
        d = float(np.max(np.abs(a - b)))
        if d != 0.0:
            raise AssertionError(
                "rsm=1.0 differs from unset under %s by %g; the inertness claim is WRONG"
                % (name, d)
            )
        inert[name] = d
    meta["rsm_one_is_inert_max_abs_diff"] = inert
    print("fact 1 OK: rsm=1.0 is byte-identical to unset (incl. under drawing samplers)")

    # ---- Fact 2: the parity targets ------------------------------------------
    seen = {}
    for r in RSM_VALUES:
        p = preds(rsm=r)
        tag = ("%g" % r).replace(".", "p")
        np.save(os.path.join(OUT_DIR, "preds_rsm_%s.npy" % tag), p)
        for prev_r, prev_p in seen.items():
            if float(np.max(np.abs(p - prev_p))) == 0.0:
                raise AssertionError(
                    "rsm=%g and rsm=%g produce IDENTICAL models, so these cells are "
                    "VACUOUS as parity evidence" % (r, prev_r)
                )
        seen[r] = p
    meta["rsm_value_tags"] = {
        "%g" % r: ("%g" % r).replace(".", "p") for r in RSM_VALUES
    }
    print("fact 2 OK: wrote %d distinct parity targets %s"
          % (len(RSM_VALUES), list(RSM_VALUES)))

    # ---- Fact 3: trees end early when a level selects nothing -----------------
    m_early = fit(rsm=EARLY_STOP_RSM)
    counts = tree_split_counts(m_early)
    if all(c == BASE["depth"] for c in counts):
        raise AssertionError(
            "rsm=%g produced only full-depth trees %s, so the early-stop rule is "
            "UNTESTED by this fixture" % (EARLY_STOP_RSM, counts)
        )
    p_early = np.asarray(
        m_early.predict(x_eval, prediction_type="RawFormulaVal"), dtype=np.float64
    )
    np.save(os.path.join(OUT_DIR, "preds_rsm_early_stop.npy"), p_early)
    meta["early_stop_tree_split_counts"] = counts
    meta["full_depth"] = BASE["depth"]
    print("fact 3 OK: rsm=%g yields short trees, per-tree split counts = %s"
          % (EARLY_STOP_RSM, counts))

    # Also record the full-depth counts so the test can assert the CONTRAST.
    meta["rsm_one_tree_split_counts"] = tree_split_counts(fit(rsm=1.0))

    # ---- Fact 4: composition with the other RNG consumers ---------------------
    composed = {}
    for name, knobs in COMPOSITION_CELLS.items():
        p = preds(rsm=0.5, **knobs)
        np.save(os.path.join(OUT_DIR, "preds_rsm_0p5_%s.npy" % name), p)
        base_p = preds(rsm=1.0, **knobs)
        d = float(np.max(np.abs(p - base_p)))
        if d == 0.0:
            raise AssertionError(
                "rsm=0.5 under %s coincides with rsm=1.0, so this cell is VACUOUS" % name
            )
        composed[name] = d
    meta["composition_vs_rsm_one_max_abs_diff"] = composed
    print("fact 4 OK: rsm composes with %s" % ", ".join(COMPOSITION_CELLS))

    # ---- Refusals -------------------------------------------------------------
    rejections = {}
    for bad in (0.0, -0.1, 1.5):
        try:
            fit(rsm=bad)
            raise AssertionError("rsm=%g must be refused by upstream" % bad)
        except AssertionError:
            raise
        except Exception as exc:
            rejections["%g" % bad] = " ".join(str(exc).split())
    meta["out_of_range_rejections"] = rejections
    print("refusals OK: rsm out of (0, 1] rejected -- %s"
          % list(rejections.values())[0][:90])

    # `colsample_bylevel` is the same parameter.
    a = preds(rsm=0.5)
    b = preds(colsample_bylevel=0.5)
    d = float(np.max(np.abs(a - b)))
    if d != 0.0:
        raise AssertionError("colsample_bylevel must alias rsm exactly; got %g" % d)
    meta["colsample_bylevel_alias_max_abs_diff"] = d
    print("alias OK: colsample_bylevel == rsm")

    with open(os.path.join(OUT_DIR, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    print("\nwrote %s" % OUT_DIR)


if __name__ == "__main__":
    main()
