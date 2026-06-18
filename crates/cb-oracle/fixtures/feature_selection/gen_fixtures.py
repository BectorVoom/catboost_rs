#!/usr/bin/env python3
"""Generate the FEAT-05 recursive-feature-selection oracle fixture from catboost
1.2.10 (OFFLINE; NEVER run in CI — CI only READS the committed artifacts).

Produces, under crates/cb-oracle/fixtures/feature_selection/:
  - X.npy [N, F] float32, y.npy [N] float64  (the shared dataset)
  - model.json                               (the FULL-feature catboost model;
                                               the Rust trainer reads its
                                               float_feature_borders for the
                                               per-step retrains)
  - config.json                              (params + the captured
                                               {selected_features,
                                               eliminated_features} partition for
                                               BOTH the ShapValues and the
                                               PredictionValuesChange (FeatureEffect)
                                               backends)

Isolating params (RESEARCH "simplest isolating params"): RMSE, boosting_type=Plain,
bootstrap_type=No, random_strength=0, permutation_count=1, thread_count=1, pinned
random_seed — so the inner per-step retrains have a deterministic draw stream and
the discrete partition is reproducible. steps == (num_for_select - num_to_select)
so EXACTLY one feature is eliminated per step (the SHAP iterative-within-step path
and a single-batch elimination then coincide bit-for-bit).

The dataset is constructed so the feature relevance ordering is UNAMBIGUOUS:
features 0,1 are the strong signal, feature 2 is weak, feature 3 is pure noise —
so the eliminated set is stable under the small border-quantization differences
between catboost's per-subset borders and the Rust trainer's sliced full-feature
borders.
"""
import json
import os

import numpy as np
from catboost import CatBoostRegressor, Pool

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 0
N = 200
F = 4

COMMON_PARAMS = dict(
    loss_function="RMSE",
    iterations=20,
    depth=4,
    learning_rate=0.1,
    l2_leaf_reg=3.0,
    random_strength=0.0,
    bootstrap_type="No",
    boosting_type="Plain",
    boost_from_average=True,
    leaf_estimation_method="Gradient",
    leaf_estimation_iterations=1,
    random_seed=SEED,
    thread_count=1,
    verbose=False,
)


def build_dataset():
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((N, F)).astype(np.float32)
    # Strong: 0,1 ; weak: 2 ; noise: 3.
    y = (
        3.0 * X[:, 0]
        + 2.0 * X[:, 1]
        + 0.3 * X[:, 2]
        + 0.0 * X[:, 3]
        + 0.05 * rng.standard_normal(N)
    ).astype(np.float64)
    return X, y


def run_select(X, y, algorithm, num_to_select):
    pool = Pool(data=X, label=y)
    model = CatBoostRegressor(**COMMON_PARAMS)
    features_for_select = list(range(F))
    steps = len(features_for_select) - num_to_select  # 1 feature per step
    summary = model.select_features(
        pool,
        features_for_select=features_for_select,
        num_features_to_select=num_to_select,
        algorithm=algorithm,
        steps=steps,
        train_final_model=False,
    )
    return {
        "selected_features": [int(i) for i in summary["selected_features"]],
        "eliminated_features": [int(i) for i in summary["eliminated_features"]],
        "steps": int(steps),
        "num_features_to_select": int(num_to_select),
    }


def per_feature_borders(X):
    """Deterministic per-feature quantile border grid (31 interior quantiles),
    saved for the Rust trainer's per-step retrains. EVERY feature gets a border
    set (the FULL-feature catboost model discards split-less features, so its own
    float_feature_borders cannot score the candidate columns 2/3 — hence an
    independent, complete grid). The discrete partition is dictated by the signal
    structure (coef 3,2,0.3,0), robust to the exact border placement."""
    Xd = X.astype(np.float64)
    borders = []
    for f in range(Xd.shape[1]):
        qs = np.quantile(Xd[:, f], np.linspace(0.0, 1.0, 33)[1:-1])
        qs = np.unique(np.round(qs, 6)).tolist()
        borders.append([float(b) for b in qs])
    return borders


def main():
    X, y = build_dataset()
    np.save(os.path.join(HERE, "X.npy"), X)
    np.save(os.path.join(HERE, "y.npy"), y)

    # The FULL-feature model (kept for provenance / inspection only).
    full = CatBoostRegressor(**COMMON_PARAMS)
    full.fit(Pool(data=X, label=y))
    full.save_model(os.path.join(HERE, "model.json"), format="json")

    num_to_select = 2  # keep the 2 strong features; drop the weak + the noise
    shap = run_select(X, y, "RecursiveByShapValues", num_to_select)
    pvc = run_select(X, y, "RecursiveByPredictionValuesChange", num_to_select)

    config = {
        "scenario": "feature_selection",
        "requirement": "FEAT-05",
        "catboost_version": "1.2.10",
        "seed": SEED,
        "thread_count": 1,
        "n_rows": N,
        "n_features": F,
        "features_for_select": list(range(F)),
        "feature_borders": per_feature_borders(X),
        "params": {k: v for k, v in COMMON_PARAMS.items()},
        "shap_values": shap,
        "prediction_values_change": pvc,
        "note": (
            "FEAT-05 (D-6.6-03) recursive feature selection partition oracle. "
            "Generated OFFLINE with catboost 1.2.10 (thread_count=1, seed pinned, "
            "Plain/No-bootstrap/random_strength=0). steps == features-to-eliminate "
            "so exactly ONE feature is eliminated per step (SHAP iterative-within-"
            "step and a single-batch elimination coincide). The Rust oracle trains "
            "the same config via cb_train::train and ranks via the Gate-C cb-model "
            "importances, asserting set/order equality of {selected,eliminated}. "
            "NEVER run in CI; CI reads the committed artifacts only."
        ),
        "artifacts": ["X.npy", "y.npy", "model.json", "config.json"],
    }
    with open(os.path.join(HERE, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)

    print("ShapValues partition:", shap)
    print("PredictionValuesChange partition:", pvc)


if __name__ == "__main__":
    main()
