"""Freeze the `final_ctr_computation_mode` / `ctr_history_unit` parity fixture.

# final_ctr_computation_mode (Default / Skip)

Measured against catboost 1.2.10 on a 25-level categorical corpus: `Skip`
returns BYTE-IDENTICAL trees, leaves, bias and the same 8 CTR splits. The ONLY
difference is that `ctr_data` comes back EMPTY.

That leaves a model whose CTR splits have no tables to look up. **catboost
1.2.10 SEGFAULTS when such a model is applied** (reproduced: the fit succeeds,
`predict` dumps core). So there is no prediction vector to freeze for `Skip` --
what this fixture pins instead is:

  * the Default predictions (the appliable model), and
  * the STRUCTURAL fact that Skip changes only `ctr_data`, captured from the
    model JSON.

The Rust port keeps training identical and refuses the APPLY with a typed error
rather than crashing -- a deliberate, documented improvement on upstream.

# ctr_history_unit (Sample / Group)

Upstream does not implement this on CPU AT ALL:

    json_helper.h:185: Error: change of option ctr_history_unit is
    unimplemented for task type CPU and was not default in previous run

So the only correct CPU behaviour is to refuse a non-default value, and the
rejection message is the contract. That is captured here rather than a
prediction vector.

Run:  python3 crates/cb-oracle/generator/gen_ctr_mode_fixtures.py
"""

import json
import os
import tempfile

import numpy as np
from catboost import CatBoostRegressor, Pool

CATBOOST_VERSION = "1.2.10"

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "fixtures")
OUT_DIR = os.path.join(FIXTURES, "ctr_modes")

SEED = 20260814
N_TRAIN = 400
N_LEVELS = 25

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
    border_count=32,
    one_hot_max_size=2,
    max_ctr_complexity=1,
    # PIN simple_ctr to the SINGLE description this crate models. Upstream's CPU
    # default is a LIST of two (`[Borders(0/1, 0.5/1, 1/1), Counter(0/1)]`,
    # catboost_options.cpp:439-453), which catboost-rs deliberately does not
    # represent (documented KNOWN PARITY GAP in catboost-rs-py/src/params.rs).
    # Leaving it at the default makes this fixture compare a 2-description model
    # against a 1-description one and it diverges by ~1.9 -- nothing to do with
    # final_ctr_computation_mode.
    simple_ctr="Borders:Prior=0.5",
)


def build():
    rng = np.random.default_rng(SEED)
    cats = ["c%02d" % v for v in rng.integers(0, N_LEVELS, N_TRAIN)]
    level = np.array([int(c[1:]) for c in cats], dtype=np.float64)
    xn = rng.normal(size=(N_TRAIN, 2))
    # Target depends on the CATEGORY LEVEL, so a CTR column is worth building.
    y = (0.5 * level + 2.0 * xn[:, 0] - xn[:, 1]).astype(np.float64)
    rows = [[cats[i], float(xn[i, 0]), float(xn[i, 1])] for i in range(N_TRAIN)]
    return cats, xn, y, rows


def eval_rows():
    r = np.random.default_rng(SEED + 1)
    cats = ["c%02d" % v for v in r.integers(0, N_LEVELS, 12)]
    xn = r.normal(size=(12, 2))
    return cats, xn, [[cats[i], float(xn[i, 0]), float(xn[i, 1])] for i in range(12)]


def model_json(m):
    path = tempfile.mktemp(suffix=".json")
    m.save_model(path, format="json")
    return json.load(open(path))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cats, xn, y, rows = build()
    ecats, exn, erows = eval_rows()

    # Persist the corpus in a Rust-loadable form: the categorical column as text,
    # the numerics as a float matrix.
    np.save(os.path.join(OUT_DIR, "X_num.npy"), np.ascontiguousarray(xn, dtype=np.float64))
    np.save(os.path.join(OUT_DIR, "y.npy"), y)
    np.save(os.path.join(OUT_DIR, "X_eval_num.npy"),
            np.ascontiguousarray(exn, dtype=np.float64))
    with open(os.path.join(OUT_DIR, "cats.txt"), "w") as fh:
        fh.write("\n".join(cats) + "\n")
    with open(os.path.join(OUT_DIR, "cats_eval.txt"), "w") as fh:
        fh.write("\n".join(ecats) + "\n")

    meta = {
        "scenario": "ctr_modes",
        "catboost_version": CATBOOST_VERSION,
        "seed": SEED,
        "params": PARAMS,
        "n_levels": N_LEVELS,
    }

    pool = Pool(rows, y, cat_features=[0])
    epool = Pool(erows, np.zeros(12), cat_features=[0])

    default = CatBoostRegressor(**PARAMS)
    default.fit(pool)
    preds = np.asarray(default.predict(epool), dtype=np.float64)
    np.save(os.path.join(OUT_DIR, "preds_Default.npy"), preds)
    print("Default preds[:3] =", np.round(preds, 6)[:3])

    skip = CatBoostRegressor(final_ctr_computation_mode="Skip", **PARAMS)
    skip.fit(pool)
    print("Skip fit ok (NOT predicted -- catboost 1.2.10 segfaults on such a model)")

    a, b = model_json(default), model_json(skip)

    def ctr_split_count(doc):
        return sum(
            1
            for t in doc.get("oblivious_trees", [])
            for s in t.get("splits", [])
            if "ctr_target_border_idx" in s or s.get("split_type") == "OnlineCtr"
        )

    same_trees = json.dumps(a.get("oblivious_trees")) == json.dumps(b.get("oblivious_trees"))
    meta["final_ctr_computation_mode"] = {
        "trees_identical": same_trees,
        "scale_and_bias_identical": a.get("scale_and_bias") == b.get("scale_and_bias"),
        "ctr_split_count": ctr_split_count(a),
        "ctr_data_keys_Default": sorted((a.get("ctr_data", {}) or {}).keys()),
        "ctr_data_keys_Skip": sorted((b.get("ctr_data", {}) or {}).keys()),
        "note": "Skip changes ONLY ctr_data; catboost 1.2.10 SEGFAULTS applying it.",
    }
    if not same_trees:
        raise AssertionError(
            "Skip changed the trees; the documented 'training is unaffected' claim "
            "no longer holds"
        )
    if (b.get("ctr_data", {}) or {}):
        raise AssertionError("Skip still produced ctr_data; it must be empty")
    if ctr_split_count(a) == 0:
        raise AssertionError(
            "the corpus produced NO CTR splits, so this fixture cannot exercise "
            "final_ctr_computation_mode at all"
        )
    print("Skip: trees identical=%s, ctr_splits=%d, ctr_data emptied=%s"
          % (same_trees, ctr_split_count(a), not (b.get("ctr_data", {}) or {})))

    # ctr_history_unit: capture the upstream CPU refusal.
    try:
        CatBoostRegressor(ctr_history_unit="Group", **PARAMS).fit(pool)
        raise AssertionError("ctr_history_unit=Group must be refused on CPU")
    except AssertionError:
        raise
    except Exception as exc:
        meta["ctr_history_unit_group_rejection"] = " ".join(str(exc).split())
        print("ctr_history_unit=Group refused:",
              meta["ctr_history_unit_group_rejection"][:110])

    with open(os.path.join(OUT_DIR, "config.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    print("wrote", OUT_DIR)


if __name__ == "__main__":
    main()
