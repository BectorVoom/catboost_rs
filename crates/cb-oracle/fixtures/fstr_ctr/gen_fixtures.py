#!/usr/bin/env python3
"""Offline fixture generator for FSTR-01 (`interaction()` /
`prediction_values_change()` CTR-aware attribution, FIC-02/FIC-03).

Generates, from catboost==1.2.10, upstream `get_feature_importance` ground
truth for a model trained on a MIXED float + categorical dataset, so the
oracle exercises the combined flat-index space (SPEC §4): 2 float feature
columns FIRST (positions [0, 1]), then 2 categorical feature columns
(positions [2, 3], `cat_features=[2, 3]`) — this float-columns-before-cat-
columns ordering is a LOAD-BEARING invariant this fixture MUST preserve (SPEC
§4's explicit scoping limitation: the Rust-side combined flat index only
coincides with upstream's true external feature index under this exact
construction).

Trains with the SAME isolating tensor-CTR parameter FAMILY as `tensor_ctr_e2e`
(`crates/cb-oracle/generator/gen_fixtures.py::gen_tensor_ctr_e2e` /
`crates/cb-train/tests/tensor_ctr_e2e_oracle_test.rs::tensor_ctr_params`):
boosting_type=Plain, one_hot_max_size=1, max_ctr_complexity=2,
simple_ctr=["Borders:Prior=0.5"], combinations_ctr=["Borders:Prior=0.5"],
learning_rate=0.1, l2_leaf_reg=3.0, random_seed=0, thread_count=1,
loss_function="Logloss" — depth/iterations are adjusted from that precedent's
depth=2/iterations=5 (empirically, per PLAN.md T4's own "this is empirical,
not guaranteed by parameters alone" note — see ISOLATING_PARAMS' comment)
because a plain additive label needs a much stronger combination signal
before the grower ever selects a GENUINE 2-feature combination CTR over two
independent simple CTRs.

HARD GATE (SPEC §7 / PLAN.md T4 — not a soft note): the generated model MUST
contain >= 1 `OnlineCtr` split whose projection spans BOTH cat features (a
genuine combination CTR). This script asserts that directly against the saved
`model.json` before writing any fixture file — if a future regeneration loses
the combination split, this script FAILS LOUDLY (non-zero exit) rather than
silently committing a degraded fixture.

Run (from repo root), with catboost==1.2.10 available:
    <py3.12-venv>/bin/python crates/cb-oracle/fixtures/fstr_ctr/gen_fixtures.py

Pinned seed, thread_count=1, bootstrap_type="No". No fabrication — every
vector comes straight from `get_feature_importance(...)`.
"""
import json
import os

import numpy as np
import pandas as pd
import catboost as cb

HERE = os.path.dirname(os.path.abspath(__file__))

SEED = 0
CATBOOST_VERSION = "1.2.10"
N_ROWS = 200
N_FLOAT = 2
N_CAT = 2
CAT_CARDINALITIES = [5, 4]  # both > one_hot_max_size=1 (genuine combination forms)

ISOLATING_PARAMS = {
    "boosting_type": "Plain",
    "one_hot_max_size": 1,
    "max_ctr_complexity": 2,
    "simple_ctr": ["Borders:Prior=0.5"],
    "combinations_ctr": ["Borders:Prior=0.5"],
    "permutation_count": 1,
    "fold_len_multiplier": 2.0,
    "counter_calc_method": "SkipTest",
    # depth=3 / iterations=15 (NOT tensor_ctr_e2e's depth=2/iterations=5 — an
    # empirical adjustment, per PLAN.md T4's "this is empirical, not
    # guaranteed by parameters alone" note): a local param sweep confirmed
    # this is the smallest depth/iteration combination (given the label's
    # float + combination-CTR signal strengths below) that makes the tree
    # grower actually select BOTH a float split AND a genuine combination-CTR
    # split (not just the combination alone), so the oracle exercises the
    # float×CTR mixed cross-product too, not only the cat×cat case.
    "depth": 3,
    "iterations": 15,
    "learning_rate": 0.1,
    "l2_leaf_reg": 3.0,
    "leaf_estimation_method": "Gradient",
    "leaf_estimation_iterations": 1,
    "bootstrap_type": "No",
    "random_strength": 0,
    "random_seed": SEED,
    "thread_count": 1,
    "loss_function": "Logloss",
    "verbose": False,
    "boost_from_average": False,
}


def build_dataset():
    """A deterministic MIXED float+categorical corpus: 2 float columns at
    positions [0, 1], then 2 integer-coded categorical columns at positions
    [2, 3] (cardinalities 5 and 4, mirroring `tensor_ctr_e2e`'s isolating
    cat-cardinality choice). The label depends on both float columns AND a
    GENUINE (cat0, cat1) INTERACTION table (not merely additive per-feature
    effects) — empirically required (verified by a local param sweep before
    committing this fixture) so the tree grower's score function actually
    prefers combining the two cat features into one CTR over two independent
    simple CTRs, satisfying the HARD GATE (>= 1 combination-CTR split).
    """
    rng = np.random.default_rng(SEED)
    x_float0 = rng.normal(0.0, 1.0, size=N_ROWS)
    x_float1 = rng.normal(0.0, 1.0, size=N_ROWS)
    cat0 = rng.integers(0, CAT_CARDINALITIES[0], size=N_ROWS)
    cat1 = rng.integers(0, CAT_CARDINALITIES[1], size=N_ROWS)

    # A random, NON-additive per-(cat0, cat1)-combination effect table (no
    # decomposition into independent per-feature terms) so the label
    # genuinely needs the combined projection, not two separate simple CTRs —
    # balanced (via a local param sweep, see ISOLATING_PARAMS's depth/
    # iterations comment) against a float signal strong enough that the
    # grower ALSO selects float splits, so the fixture exercises the mixed
    # float×CTR cross-product, not only the cat×cat one.
    combo_effect_table = rng.normal(0.0, 2.5, size=CAT_CARDINALITIES)
    combo_effect = combo_effect_table[cat0, cat1]

    logit = 0.8 * x_float0 - 0.64 * x_float1 + combo_effect
    logit = logit - logit.mean()
    y = (logit > 0.0).astype(np.float64)

    # Rust-side ground-truth arrays (float64 for X_float, int32 for X_cat —
    # loadable by ndarray-npy without a str-npy dependency, A4 convention).
    x_float = np.stack([x_float0, x_float1], axis=1).astype(np.float64)
    x_cat = np.stack([cat0, cat1], axis=1).astype(np.int32)

    # The upstream-facing DataFrame: float columns FIRST, then cat columns
    # (stringified integer codes, A4 — `cb_data::stringify_int_category`'s
    # convention, `str(int)`), positions [0, 1, 2, 3] with cat_features=[2, 3]
    # (SPEC §4 load-bearing float-before-cat column-order invariant).
    df = pd.DataFrame(
        {
            0: x_float0,
            1: x_float1,
            2: cat0.astype(int).astype(str),
            3: cat1.astype(int).astype(str),
        }
    )
    return x_float, x_cat, y, df


def assert_has_combination_ctr(model_json_path):
    """HARD GATE (SPEC §7 / PLAN.md T4): the saved `model.json` must contain
    >= 1 `OnlineCtr` split whose projection spans >= 2 cat features (a
    genuine combination CTR), not just simple/single-feature CTRs.

    Upstream's `model.json` schema (verified empirically against this
    fixture's own saved output, catboost 1.2.10): each SPLIT of
    `split_type == "OnlineCtr"` carries only a `split_index` (the position
    into the model-wide CTR-split pool) — the actual combined PROJECTION
    (which cat feature(s) it tests) lives separately, at
    `features_info.ctrs[split_index].elements`, a list of
    `{"cat_feature_index": <local index>, ...}` entries. A projection with
    `len(elements) >= 2` is a genuine combination CTR.
    """
    with open(model_json_path, encoding="utf-8") as fh:
        mj = json.load(fh)

    ctr_defs = mj.get("features_info", {}).get("ctrs", [])
    if not ctr_defs:
        raise SystemExit(
            "HARD GATE FAILED: no CTR feature definitions found in "
            f"features_info.ctrs at all ({model_json_path}) — adjust "
            "cardinalities/iterations/depth."
        )
    max_projection_len = max(len(c.get("elements", [])) for c in ctr_defs)
    if max_projection_len < 2:
        raise SystemExit(
            "HARD GATE FAILED: the trained model has ONLY simple/single-feature "
            f"CTR projections (max projection length {max_projection_len}) — no "
            "genuine combination CTR was formed. Adjust cardinalities/"
            f"iterations/depth and regenerate ({model_json_path})."
        )
    print(
        f"[fstr_ctr] HARD GATE OK: found {len(ctr_defs)} CTR feature "
        f"definition(s), max projection length {max_projection_len} (>= 2, "
        "genuine combination CTR present)."
    )


def main():
    x_float, x_cat, y, df = build_dataset()
    np.save(os.path.join(HERE, "X_float.npy"), x_float, allow_pickle=False)
    np.save(os.path.join(HERE, "X_cat.npy"), x_cat, allow_pickle=False)
    np.save(os.path.join(HERE, "y.npy"), y.astype(np.float64), allow_pickle=False)

    pool = cb.Pool(df, y, cat_features=[2, 3])

    # Low-level CatBoost(params) API (not CatBoostClassifier) so
    # permutation_count is honored exactly as a training param, matching the
    # tensor_ctr_e2e precedent.
    model = cb.CatBoost(ISOLATING_PARAMS)
    model.fit(pool)

    model.save_model(os.path.join(HERE, "model.cbm"), format="cbm")
    model_json_path = os.path.join(HERE, "model.json")
    model.save_model(model_json_path, format="json")

    # HARD GATE: fail loudly, before writing any importance fixture, if the
    # trained model lacks a genuine combination CTR split.
    assert_has_combination_ctr(model_json_path)

    # --- Interaction (structural-only, no `data=` needed) -------------------
    interaction_raw = model.get_feature_importance(type="Interaction")
    # Upstream returns [[i, j, score], ...] (float-typed positions) — flatten
    # row-major to match this project's EXISTING `interaction.npy` convention
    # (flattened [feature_i, feature_j, score] triples, the SAME shape
    # `fstr_oracle_test.rs` already consumes, per its own docstring).
    interaction_arr = np.asarray(interaction_raw, dtype=np.float64)
    interaction_flat = interaction_arr.reshape(-1)
    np.save(os.path.join(HERE, "interaction.npy"), interaction_flat, allow_pickle=False)

    # --- PredictionValuesChange (needs `data=pool`) --------------------------
    pvc = np.asarray(
        model.get_feature_importance(type="PredictionValuesChange", data=pool),
        dtype=np.float64,
    )
    np.save(os.path.join(HERE, "prediction_values_change.npy"), pvc, allow_pickle=False)

    # --- Predictions (sanity gate, per PLAN.md T5's Risk note: verify the
    # Rust-lifted model's predictions match upstream BEFORE trusting an
    # interaction/PVC mismatch as an algorithm bug, not a model-loading one) --
    predictions = np.asarray(
        model.predict(df, prediction_type="RawFormulaVal"), dtype=np.float64
    )
    np.save(os.path.join(HERE, "predictions.npy"), predictions, allow_pickle=False)

    config = {
        "catboost_version": CATBOOST_VERSION,
        "scenario": "fstr_ctr",
        "requirement": "FSTR-01",
        "seed": SEED,
        "thread_count": 1,
        "n_rows": N_ROWS,
        "n_float": N_FLOAT,
        "n_cat": N_CAT,
        "cat_features": [2, 3],
        "cat_cardinalities": CAT_CARDINALITIES,
        "params": ISOLATING_PARAMS,
        "column_order_invariant": (
            "SPEC section 4 LOAD-BEARING invariant: this fixture's Pool places "
            "ALL float feature columns (positions [0, 1]) BEFORE ALL "
            "categorical feature columns (positions [2, 3], cat_features=[2, "
            "3]) so the Rust-side combined flat index (floats [0, n_float), "
            "cats [n_float, n_float+n_cat)) coincides with upstream's true "
            "external feature index. Regenerating this fixture with a "
            "DIFFERENT column order silently breaks this invariant and the "
            "AT-FIC02d/AT-FIC03d oracle comparisons that depend on it — do "
            "NOT reorder the columns without updating SPEC.md section 4."
        ),
        "note": (
            "FSTR-01: interaction() / prediction_values_change() CTR-aware "
            "attribution ground truth. Mixed float+categorical model (2 float, "
            "2 cat), trained with the tensor_ctr_e2e-family isolating params "
            "(depth/iterations empirically adjusted, see ISOLATING_PARAMS' "
            "comment, so the grower selects BOTH a float split and a genuine "
            "combination CTR, not just the combination alone). HARD GATE "
            "verified: >= 1 OnlineCtr split has a >= 2-cat-feature projection "
            "(a genuine combination CTR), not just simple CTRs. "
            "interaction.npy is flattened [feature_i, feature_j, score] "
            "triples (the existing fstr_oracle_test.rs convention); "
            "prediction_values_change.npy is length n_float+n_cat in flat-"
            "index order (floats then cats, matching the Pool's own column "
            "order, so upstream's external index numbering coincides, SPEC "
            "section 4 / section 9 risk 9)."
        ),
        "npy_schema": {
            "X_float.npy": "[N, 2] float64 — the 2 float feature columns.",
            "X_cat.npy": "[N, 2] int32 — the 2 raw categorical columns as integer codes (Rust stringifies via cb_data::stringify_int_category, A4).",
            "y.npy": "[N] float64 — Logloss labels.",
            "model.cbm": "the trained upstream catboost 1.2.10 model, binary form.",
            "model.json": "the same model, JSON form (splits + leaf_values + borders + baked ctr_data).",
            "interaction.npy": "flattened [feature_i, feature_j, score] triples from get_feature_importance(type='Interaction').",
            "prediction_values_change.npy": "[n_float+n_cat] float64 from get_feature_importance(type='PredictionValuesChange', data=pool), summing to 100.",
            "predictions.npy": "[N] float64, upstream RawFormulaVal -- a sanity gate proving the Rust-lifted model matches upstream BEFORE trusting an interaction/PVC mismatch as an algorithm bug (PLAN.md T5 Risk note).",
        },
    }
    with open(os.path.join(HERE, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True, default=float)
        fh.write("\n")

    print("interaction (first 5 rows):", interaction_arr[:5].tolist())
    print("prediction_values_change:", pvc.tolist())
    print(f"[fstr_ctr] wrote fixture artifacts under {HERE}")


if __name__ == "__main__":
    main()
