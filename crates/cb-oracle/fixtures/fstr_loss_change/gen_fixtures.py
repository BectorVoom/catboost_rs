#!/usr/bin/env python3
"""Offline fixture generator for 06.6-06 (LossFunctionChange + non-symmetric fstr).

Generates, from catboost==1.2.10 in the project `.venv`:
  - oblivious LossFunctionChange ground truth on the SAME binclf model that
    produced `model_serde/binclf/model.json` (identical params/seed/inputs), so
    the Rust `loss_function_change` backend can be oracle-locked <=1e-5.
  - a non-symmetric (Depthwise) model (`.cbm`) plus its upstream
    PredictionValuesChange / Interaction / LossFunctionChange ground truth, so
    the generalized non-symmetric PVC/Interaction recursion can be oracle-locked.

Run (from repo root):
    .venv/bin/python crates/cb-oracle/fixtures/fstr_loss_change/gen_fixtures.py

Pinned seed, thread_count=1, bootstrap_type="No". No fabrication — every vector
comes straight from `get_feature_importance(...)`.
"""
import json
import os

import numpy as np
import catboost as cb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
INPUTS = os.path.join(ROOT, "crates", "cb-oracle", "fixtures", "inputs", "numeric_tiny")


def load_binclf_pool():
    X = np.load(os.path.join(INPUTS, "X.npy"))
    y = np.load(os.path.join(INPUTS, "y.npy"))
    med = np.median(y)
    yb = (y > med).astype(int)
    return X, yb, cb.Pool(X, yb)


def gen_oblivious_lfc():
    """LossFunctionChange on the canonical binclf oblivious model."""
    X, yb, pool = load_binclf_pool()
    m = cb.CatBoostClassifier(
        boost_from_average=False, bootstrap_type="No", depth=2, iterations=5,
        l2_leaf_reg=3.0, leaf_estimation_iterations=1,
        leaf_estimation_method="Gradient", learning_rate=0.1, random_seed=0,
        random_strength=0, score_function="L2", thread_count=1, verbose=False,
    )
    m.fit(pool)
    lfc = np.asarray(m.get_feature_importance(type="LossFunctionChange", data=pool),
                     dtype=np.float64)
    np.save(os.path.join(HERE, "oblivious_loss_function_change.npy"), lfc)
    # The X/y the Rust side feeds to its own apply + SHAP + Logloss reproduction.
    np.save(os.path.join(HERE, "binclf_X.npy"), X.astype(np.float64))
    np.save(os.path.join(HERE, "binclf_y.npy"), yb.astype(np.float64))
    return lfc.tolist()


def _numeric_regression_pool():
    """The `numeric_tiny` X with its RAW continuous y (regression target)."""
    X = np.load(os.path.join(INPUTS, "X.npy"))
    y = np.load(os.path.join(INPUTS, "y.npy")).astype(np.float64)
    return X, y, cb.Pool(X, y)


def gen_regression_lfc(tag: str, loss_function: str):
    """FSTR-02 (FL-04): LossFunctionChange on a float-only oblivious REGRESSOR
    trained with `loss_function` on `numeric_tiny`. Freezes the model `.cbm`,
    the X/y the Rust side feeds to its own apply + SHAP + final-error closure,
    and the upstream LossFunctionChange vector. The model's own metric
    `GetFinalError` (RMSE / MAE / MAPE / Quantile) is what the Rust closure
    must reproduce <=1e-5.

    Isolating params mirror the oblivious binclf fixture (`bootstrap_type="No"`,
    depth=2, iterations=5, l2_leaf_reg=3.0, learning_rate=0.1, seed=0,
    random_strength=0, score_function="L2", thread_count=1,
    boost_from_average=False) so the model is fully determined and the Rust
    reconstruction (`load_cbm`) is exact."""
    X, y, pool = _numeric_regression_pool()
    m = cb.CatBoostRegressor(
        loss_function=loss_function, boost_from_average=False, bootstrap_type="No",
        depth=2, iterations=5, l2_leaf_reg=3.0, leaf_estimation_iterations=1,
        leaf_estimation_method="Gradient", learning_rate=0.1, random_seed=0,
        random_strength=0, score_function="L2", thread_count=1, verbose=False,
    )
    m.fit(pool)
    m.save_model(os.path.join(HERE, f"{tag}_model.cbm"), format="cbm")
    m.save_model(os.path.join(HERE, f"{tag}_model.json"), format="json")
    lfc = np.asarray(m.get_feature_importance(type="LossFunctionChange", data=pool),
                     dtype=np.float64)
    np.save(os.path.join(HERE, f"{tag}_loss_function_change.npy"), lfc)
    np.save(os.path.join(HERE, f"{tag}_X.npy"), X.astype(np.float64))
    np.save(os.path.join(HERE, f"{tag}_y.npy"), y.astype(np.float64))
    print(f"{tag} ({loss_function}) LFC:", lfc.tolist())
    return lfc.tolist()


def gen_non_symmetric():
    """Non-symmetric Depthwise model + PVC/Interaction/LFC ground truth."""
    X, yb, pool = load_binclf_pool()
    m = cb.CatBoostClassifier(
        boost_from_average=False, bootstrap_type="No", grow_policy="Depthwise",
        max_depth=3, iterations=4, l2_leaf_reg=3.0, leaf_estimation_iterations=1,
        leaf_estimation_method="Gradient", learning_rate=0.3, random_seed=42,
        random_strength=0, score_function="L2", thread_count=1, verbose=False,
    )
    m.fit(pool)
    # .cbm carries LeafValues + LeafWeights + the non-symmetric node graph, so
    # `cb_model::load_cbm` reconstructs `non_symmetric_trees` with leaf_weights.
    m.save_model(os.path.join(HERE, "non_symmetric_model.cbm"), format="cbm")
    m.save_model(os.path.join(HERE, "non_symmetric_model.json"), format="json")

    pvc = np.asarray(m.get_feature_importance(type="PredictionValuesChange"),
                     dtype=np.float64)
    inter = np.asarray(m.get_feature_importance(type="Interaction"),
                       dtype=np.float64)  # rows of [i, j, score]
    lfc = np.asarray(m.get_feature_importance(type="LossFunctionChange", data=pool),
                     dtype=np.float64)
    np.save(os.path.join(HERE, "non_symmetric_pvc.npy"), pvc)
    np.save(os.path.join(HERE, "non_symmetric_interaction.npy"),
            inter.reshape(-1).astype(np.float64))
    np.save(os.path.join(HERE, "non_symmetric_loss_function_change.npy"), lfc)
    return pvc.tolist(), inter.tolist(), lfc.tolist()


def main():
    obl_lfc = gen_oblivious_lfc()
    ns_pvc, ns_inter, ns_lfc = gen_non_symmetric()
    # FSTR-02 (FL-04a/FL-04b): per-numeric-loss oblivious regressor LFC fixtures.
    rmse_lfc = gen_regression_lfc("rmse", "RMSE")
    mae_lfc = gen_regression_lfc("mae", "MAE")
    mape_lfc = gen_regression_lfc("mape", "MAPE")
    quantile_lfc = gen_regression_lfc("quantile", "Quantile:alpha=0.5")
    config = {
        "catboost_version": "1.2.10",
        "input_dataset": "numeric_tiny",
        "scenario": "fstr_loss_change",
        "thread_count": 1,
        "seed_oblivious": 0,
        "seed_non_symmetric": 42,
        "seed_regression": 0,
        "note": (
            "06.6-06: LossFunctionChange (MODEL-03/D-12) + non-symmetric "
            "PVC/Interaction (D-6.6-10). FSTR-02 (FL-04): per-loss oblivious "
            "REGRESSOR LFC (RMSE/MAE/MAPE/Quantile:alpha=0.5). All vectors from "
            "catboost 1.2.10 get_feature_importance(...). Oblivious binclf model "
            "= the binclf params; non-symmetric model = Depthwise max_depth=3; "
            "regression models = the binclf isolating params on the raw y."
        ),
        "oblivious_loss_function_change": obl_lfc,
        "non_symmetric_pvc": ns_pvc,
        "non_symmetric_interaction": ns_inter,
        "non_symmetric_loss_function_change": ns_lfc,
        "rmse_loss_function_change": rmse_lfc,
        "mae_loss_function_change": mae_lfc,
        "mape_loss_function_change": mape_lfc,
        "quantile_loss_function_change": quantile_lfc,
        "artifacts": [
            "oblivious_loss_function_change.npy",
            "binclf_X.npy",
            "binclf_y.npy",
            "non_symmetric_model.cbm",
            "non_symmetric_model.json",
            "non_symmetric_pvc.npy",
            "non_symmetric_interaction.npy",
            "non_symmetric_loss_function_change.npy",
        ] + [
            f"{tag}_{suffix}"
            for tag in ("rmse", "mae", "mape", "quantile")
            for suffix in ("model.cbm", "model.json", "loss_function_change.npy",
                           "X.npy", "y.npy")
        ],
    }
    with open(os.path.join(HERE, "config.json"), "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    print("oblivious LFC:", obl_lfc)
    print("non-sym PVC:", ns_pvc)
    print("non-sym Interaction:", ns_inter)
    print("non-sym LFC:", ns_lfc)


if __name__ == "__main__":
    main()
