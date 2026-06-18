#!/usr/bin/env python3
"""Offline fixture generator for 06.6-07 (advanced SHAP-family fstr: MODEL-05).

Generates, from catboost==1.2.10 in the project `.venv`:
  - ShapInteractionValues ground truth (n_obj, n_feat+1, n_feat+1) on the SAME
    binclf oblivious model that produced `model_serde/binclf/model.json`
    (identical params/seed/inputs), plus the regular ShapValues so the bias-slot
    index convention (RESEARCH Open Question 2) can be reverse-mapped empirically.
  - PredictionDiff ground truth (n_features,) on X[:2].
  - SAGE / SageValues ground truth (n_features,) with a PINNED random_seed
    (RESEARCH gate 2: deterministic + seed-reproducible — verified across seeds).
  - A NON-SYMMETRIC (Depthwise) model `.cbm` + json plus its regular ShapValues
    AND its ShapInteractionValues, so the generalized non-symmetric TreeSHAP
    (D-6.6-10) can be oracle-locked >=1 non-symmetric case.

Run (from repo root):
    .venv/bin/python crates/cb-oracle/fixtures/advanced_fstr/gen_fixtures.py

Pinned seeds, thread_count=1, bootstrap_type="No". No fabrication — every vector
comes straight from `get_feature_importance(...)`.
"""
import json
import os

import numpy as np
import catboost as cb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
INPUTS = os.path.join(ROOT, "crates", "cb-oracle", "fixtures", "inputs", "numeric_tiny")

SAGE_SEED = 0  # pinned; SAGE is deterministic + seed-reproducible (RESEARCH gate 2)


def load_binclf_pool():
    X = np.load(os.path.join(INPUTS, "X.npy"))
    y = np.load(os.path.join(INPUTS, "y.npy"))
    med = np.median(y)
    yb = (y > med).astype(int)
    return X, yb, cb.Pool(X, yb)


def binclf_model():
    """The canonical binclf oblivious model (matches model_serde/binclf)."""
    _, _, pool = load_binclf_pool()
    m = cb.CatBoostClassifier(
        boost_from_average=False, bootstrap_type="No", depth=2, iterations=5,
        l2_leaf_reg=3.0, leaf_estimation_iterations=1,
        leaf_estimation_method="Gradient", learning_rate=0.1, random_seed=0,
        random_strength=0, score_function="L2", thread_count=1, verbose=False,
    )
    m.fit(pool)
    return m


def gen_oblivious_advanced():
    X, yb, pool = load_binclf_pool()
    m = binclf_model()

    shap = np.asarray(m.get_feature_importance(type="ShapValues", data=pool),
                      dtype=np.float64)  # (n_obj, n_feat+1)
    inter = np.asarray(
        m.get_feature_importance(type="ShapInteractionValues", data=pool),
        dtype=np.float64)  # (n_obj, n_feat+1, n_feat+1)
    pdiff = np.asarray(
        m.get_feature_importance(type="PredictionDiff", data=X[:2]),
        dtype=np.float64)  # (n_features,)
    sage = np.asarray(
        m.get_feature_importance(type="SageValues", data=pool),
        dtype=np.float64)  # (n_features,)

    np.save(os.path.join(HERE, "oblivious_shap.npy"), shap)
    np.save(os.path.join(HERE, "oblivious_shap_interaction.npy"),
            inter.reshape(-1))
    np.save(os.path.join(HERE, "oblivious_shap_interaction_shape.npy"),
            np.asarray(inter.shape, dtype=np.int64))
    np.save(os.path.join(HERE, "prediction_diff.npy"), pdiff)
    np.save(os.path.join(HERE, "sage_values.npy"), sage)
    np.save(os.path.join(HERE, "binclf_X.npy"), X.astype(np.float64))
    np.save(os.path.join(HERE, "binclf_y.npy"), yb.astype(np.float64))

    # --- Reverse-map the +1 bias slot (RESEARCH Open Question 2) -------------
    # ShapInteractionValues is symmetric in (i, j); the diagonal+offdiag for a
    # feature collapses (row-sum over j) to that feature's ShapValue, and the
    # bias slot (last index) carries the expected-value term. Verify:
    #   sum_j inter[obj, i, j] == shap[obj, i]   for all i (incl. bias slot).
    n_obj, m1, m2 = inter.shape
    assert m1 == m2 == shap.shape[1], (m1, m2, shap.shape)
    rowsum = inter.sum(axis=2)  # (n_obj, n_feat+1)
    max_dev = float(np.max(np.abs(rowsum - shap)))
    bias_idx = shap.shape[1] - 1
    print(f"[oblivious] ShapInteraction row-sum vs ShapValues max|dev|={max_dev:.3e}; "
          f"bias slot = last index {bias_idx} (n_feat+1={shap.shape[1]})")

    return shap.tolist(), list(inter.shape), pdiff.tolist(), sage.tolist(), max_dev


def gen_non_symmetric_advanced():
    X, yb, pool = load_binclf_pool()
    m = cb.CatBoostClassifier(
        boost_from_average=False, bootstrap_type="No", grow_policy="Depthwise",
        max_depth=3, iterations=4, l2_leaf_reg=3.0, leaf_estimation_iterations=1,
        leaf_estimation_method="Gradient", learning_rate=0.3, random_seed=42,
        random_strength=0, score_function="L2", thread_count=1, verbose=False,
    )
    m.fit(pool)
    m.save_model(os.path.join(HERE, "non_symmetric_model.cbm"), format="cbm")
    m.save_model(os.path.join(HERE, "non_symmetric_model.json"), format="json")

    shap = np.asarray(m.get_feature_importance(type="ShapValues", data=pool),
                      dtype=np.float64)  # (n_obj, n_feat+1)
    inter = np.asarray(
        m.get_feature_importance(type="ShapInteractionValues", data=pool),
        dtype=np.float64)
    np.save(os.path.join(HERE, "non_symmetric_shap.npy"), shap.reshape(-1))
    np.save(os.path.join(HERE, "non_symmetric_shap_interaction.npy"),
            inter.reshape(-1))
    np.save(os.path.join(HERE, "non_symmetric_shap_interaction_shape.npy"),
            np.asarray(inter.shape, dtype=np.int64))

    rowsum = inter.sum(axis=2)
    max_dev = float(np.max(np.abs(rowsum - shap)))
    print(f"[non-sym] ShapInteraction row-sum vs ShapValues max|dev|={max_dev:.3e}")
    return shap.tolist(), list(inter.shape), max_dev


def verify_sage_reproducible():
    """RESEARCH gate 2: SAGE is deterministic + seed-reproducible."""
    _, _, pool = load_binclf_pool()
    a = binclf_model().get_feature_importance(type="SageValues", data=pool)
    b = binclf_model().get_feature_importance(type="SageValues", data=pool)
    same = bool(np.allclose(np.asarray(a), np.asarray(b)))
    print(f"[sage] same-seed reproducible: {same}")
    return same


def main():
    obl_shap, obl_inter_shape, pdiff, sage, obl_dev = gen_oblivious_advanced()
    ns_shap, ns_inter_shape, ns_dev = gen_non_symmetric_advanced()
    sage_repro = verify_sage_reproducible()
    config = {
        "catboost_version": "1.2.10",
        "input_dataset": "numeric_tiny",
        "scenario": "advanced_fstr",
        "thread_count": 1,
        "seed_oblivious": 0,
        "seed_non_symmetric": 42,
        "sage_random_seed": SAGE_SEED,
        "bias_slot_index": "last (n_feat+1 - 1); row-sum over j collapses to ShapValues",
        "shap_interaction_rowsum_dev_oblivious": obl_dev,
        "shap_interaction_rowsum_dev_non_symmetric": ns_dev,
        "sage_same_seed_reproducible": sage_repro,
        "note": (
            "06.6-07: ShapInteractionValues / PredictionDiff / SAGE (MODEL-05) + "
            "non-symmetric TreeSHAP (D-6.6-10). All vectors from catboost 1.2.10 "
            "get_feature_importance(...). Oblivious model = the binclf params; "
            "non-symmetric model = Depthwise max_depth=3. SAGE seed-pinned, "
            "deterministic (D-6.6-11 fallback a)."
        ),
        "oblivious_shap_interaction_shape": obl_inter_shape,
        "non_symmetric_shap_interaction_shape": ns_inter_shape,
        "prediction_diff": pdiff,
        "sage_values": sage,
        "artifacts": [
            "oblivious_shap.npy",
            "oblivious_shap_interaction.npy",
            "oblivious_shap_interaction_shape.npy",
            "prediction_diff.npy",
            "sage_values.npy",
            "binclf_X.npy",
            "binclf_y.npy",
            "non_symmetric_model.cbm",
            "non_symmetric_model.json",
            "non_symmetric_shap.npy",
            "non_symmetric_shap_interaction.npy",
            "non_symmetric_shap_interaction_shape.npy",
        ],
    }
    with open(os.path.join(HERE, "config.json"), "w") as f:
        json.dump(config, f, indent=2, sort_keys=True, default=float)
    print("prediction_diff:", pdiff)
    print("sage_values:", sage)


if __name__ == "__main__":
    main()
