"""Freeze the `class_names` parity facts from catboost 1.2.10.

`class_names` maps arbitrary class labels onto class INDICES, and the ORDER given is
the parameter's whole point: it decides which label is the positive class and how
`predict_proba`'s columns are arranged. Both orders are frozen so the reversal is
testable, not merely asserted.

EVERY default the two sides disagree on is pinned explicitly. Leaving them unset made
this comparison read 0.177 max|diff| and 58/60 label agreement -- which looks like a
`class_names` bug and is not one; with the defaults pinned it is 0.0 and 60/60. Same
trap as the cv/ORCH-01 fixture: pin on BOTH sides.

Run:  python3 crates/cb-oracle/generator/gen_class_names_fixtures.py
"""

import json
import os

import numpy as np
from catboost import CatBoostClassifier

CATBOOST_VERSION = "1.2.10"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "fixtures", "class_names")

SEED = 3
N, N_FEATURES = 60, 3
ORDERS = [["neg", "pos"], ["pos", "neg"]]

PARAMS = dict(
    iterations=5,
    depth=2,
    learning_rate=0.3,
    l2_leaf_reg=3.0,
    random_seed=0,
    bootstrap_type="No",
    random_strength=0,
    leaf_estimation_method="Gradient",
    score_function="L2",
    leaf_estimation_iterations=1,
    border_count=32,
    boost_from_average=False,
    grow_policy="SymmetricTree",
    thread_count=1,
    verbose=False,
)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    x = np.ascontiguousarray(rng.normal(size=(N, N_FEATURES)), dtype=np.float32)
    y01 = (2 * x[:, 0] - x[:, 1] > 0).astype(np.int64)
    labels = np.array(["neg", "pos"])[y01]

    np.save(os.path.join(OUT_DIR, "X.npy"), x)
    np.save(os.path.join(OUT_DIR, "y_index.npy"), y01.astype(np.float32))

    meta = {
        "scenario": "class_names",
        "catboost_version": CATBOOST_VERSION,
        "seed": SEED,
        "params": PARAMS,
        "orders": ORDERS,
        "labels": ["neg", "pos"],
    }

    probas = {}
    for order in ORDERS:
        m = CatBoostClassifier(**PARAMS, class_names=order)
        m.fit(x, labels)
        pred = np.asarray(m.predict(x)).ravel().astype(str)
        proba = np.asarray(m.predict_proba(x), dtype=np.float64)
        stem = "_".join(order)
        np.save(os.path.join(OUT_DIR, f"proba_{stem}.npy"), proba)
        # Predicted labels as class INDICES under THIS order, so the fixture is
        # numeric and the test does the label mapping itself.
        idx = np.array([order.index(p) for p in pred], dtype=np.float32)
        np.save(os.path.join(OUT_DIR, f"pred_index_{stem}.npy"), idx)
        probas[stem] = proba
        assert list(m.classes_) == order, "classes_ must follow the given order"

    # Vacuity guard: the two orders must actually DIFFER, or the reversal test below
    # would pass against an implementation that ignored the order entirely.
    a, b = probas["neg_pos"], probas["pos_neg"]
    swapped = float(np.max(np.abs(a - b[:, ::-1])))
    straight = float(np.max(np.abs(a - b)))
    if straight == 0.0:
        raise AssertionError(
            "the two class_names orders produced IDENTICAL probability columns; the "
            "order is not being honoured, so this fixture cannot test the reversal"
        )
    meta["reversed_columns_max_abs_diff"] = swapped
    meta["straight_columns_max_abs_diff"] = straight
    print("orders differ (straight max|diff| = %.6g)" % straight)
    print("reversing the columns aligns them (max|diff| = %.3g)" % swapped)

    with open(os.path.join(OUT_DIR, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    print("wrote", OUT_DIR)


if __name__ == "__main__":
    main()
