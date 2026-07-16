"""Generate the CTR model-loading oracle fixtures (Phase 23, CTR-01..05).

FROZEN GENERATOR — the committed `simple.cbm` / `combo.cbm` / `*.json` /
`*_preds.npy` / `X.npy` / `y.npy` / `X_float.npy` / `cat_cols.json` /
`config.json` under this directory are the GROUND TRUTH `cb_model::decode_cbm`
is dissected/tested against. T3's byte-level constants (bucket counts, hashes,
int_counts, counter_denominator) are read directly off the COMMITTED
`simple.cbm`, not re-derived by running this script. CI does NOT run this
script (no `catboost` install in CI) and never regenerates these fixtures.

# Reproducibility caveat (load-bearing — do not "fix" by re-running this file)

Empirically, re-running this exact recipe (same `catboost==1.2.10` venv, same
pinned `numpy.random.RandomState(0)` seed, same hyperparameters including
`thread_count=1` / `random_seed=0`) on the SAME machine at a LATER time
produced a DIFFERENT-SIZED `.cbm` (different float-feature quantization
borders, different CTR split count) than the committed artifacts. CatBoost's
border/quantization step appears to have a source of run-to-run
nondeterminism independent of the exposed `random_seed` (observed even at
`thread_count=1`). Because of this, `gen_fixtures.py` documents the pinned
RECIPE for provenance only — it is NOT expected to byte-reproduce the
committed fixtures, and must never be used to regenerate them for CI or for
re-dissection. The checked-in `.cbm` files ARE the fixtures.

# Pinned recipe (best-effort provenance record)

  - `catboost==1.2.10`.
  - `numpy.random.RandomState(0)`.
  - `n=200` documents-per-class draws feeding `f0`/`c0`/`c1`; the actually
    committed fixtures reflect `n=400` rows with ONE float feature (`f0`) and
    two categorical features (`c0` in `{0..4}`, `c1` in `{0..3}`) — confirmed
    directly from the committed `X.npy` (shape `(400, 3)`) and
    `{simple,combo}.json` (`features_info.float_features` length 1,
    `features_info.categorical_features` length 2). This script reflects that
    ACTUAL shape (not an earlier draft), so re-running it is a best-effort
    provenance approximation, not a guaranteed byte match (see caveat above).
  - Target is CAT-DRIVEN (`y` depends on a categorical column) so upstream
    actually selects `OnlineCtr` splits — a weak/absent cat signal yields
    ZERO CTR splits and this fixture is useless for CTR-04.
  - `cat_features` = the two trailing columns of the combined feature matrix.
  - Params: `bootstrap_type="No"`, `depth=4`, `iterations=10`,
    `l2_leaf_reg=3.0`, `learning_rate=0.1`, `random_seed=0`,
    `random_strength=0`, `thread_count=1`, `loss_function="Logloss"`,
    `allow_writing_files=False`, `logging_level="Silent"`.
  - `simple`: `max_ctr_complexity=1` (SimpleCtr only, one cat feature per CTR).
  - `combo`: `max_ctr_complexity=2` (SimpleCtr + CombinationCtr, up to two cat
    features per CTR).

# What this script actually (re)produces when run

Given the reproducibility caveat, this script is kept for provenance / manual
inspection. It regenerates its own scratch `.cbm`/`.json`/`*_preds.npy` next to
(NOT overwriting) the committed fixtures, and separately derives
`X_float.npy` / `cat_cols.json` / `config.json` FROM THE COMMITTED `X.npy` and
`{simple,combo}.json` (never from a freshly trained model), so those derived
files are guaranteed internally consistent with the frozen predictions
regardless of training nondeterminism.
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def regenerate_scratch_copy():
    """Best-effort re-run of the pinned recipe into `_scratch_regen/` (NOT the
    committed fixtures — see the module-level reproducibility caveat)."""
    from catboost import CatBoostClassifier, Pool

    out_dir = os.path.join(HERE, "_scratch_regen")
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.RandomState(0)
    n = 400
    f0 = rng.normal(size=n)
    c0 = rng.randint(0, 5, size=n).astype(str)
    c1 = rng.randint(0, 4, size=n).astype(str)
    y = ((f0 + (c0.astype(int) == 2).astype(float) + rng.normal(scale=0.3, size=n)) > 0.5).astype(int)

    X = np.column_stack([f0, c0, c1])
    cat_features = [1, 2]

    common = dict(
        bootstrap_type="No",
        depth=4,
        iterations=10,
        l2_leaf_reg=3.0,
        learning_rate=0.1,
        random_seed=0,
        random_strength=0,
        thread_count=1,
        loss_function="Logloss",
        allow_writing_files=False,
        logging_level="Silent",
    )

    for tag, mcc in [("simple", 1), ("combo", 2)]:
        model = CatBoostClassifier(max_ctr_complexity=mcc, **common)
        pool = Pool(X, label=y, cat_features=cat_features)
        model.fit(pool)
        model.save_model(os.path.join(out_dir, f"{tag}.cbm"), format="cbm")
        model.save_model(os.path.join(out_dir, f"{tag}.json"), format="json")
        preds = model.predict(X, prediction_type="RawFormulaVal")
        np.save(os.path.join(out_dir, f"{tag}_preds.npy"), preds)
    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "y.npy"), y)
    print(f"scratch re-run written to {out_dir} (provenance only, not committed fixtures)")


def derive_from_committed():
    """Derive `X_float.npy` / `cat_cols.json` / `config.json` from the COMMITTED
    `X.npy` + `{simple,combo}.json` — guaranteed consistent with the frozen
    `*_preds.npy` regardless of training nondeterminism."""
    x = np.load(os.path.join(HERE, "X.npy"), allow_pickle=True)
    float_col = x[:, 0].astype(np.float64)
    np.save(os.path.join(HERE, "X_float.npy"), float_col.reshape(-1, 1))

    # A4 plain-integer stringification (`str(int(v))`) — the exact form
    # `cb_data::calc_cat_feature_hash` / `stringify_int_category` expects.
    cat0 = [str(int(float(v))) for v in x[:, 1]]
    cat1 = [str(int(float(v))) for v in x[:, 2]]
    with open(os.path.join(HERE, "cat_cols.json"), "w") as fh:
        json.dump({"c0": cat0, "c1": cat1}, fh)

    config = {
        "n_rows": int(x.shape[0]),
        "n_float_features": 1,
        "cat_features_columns": ["c0", "c1"],
        "scenarios": {},
    }
    for tag in ("simple", "combo"):
        with open(os.path.join(HERE, f"{tag}.json")) as fh:
            model_json = json.load(fh)
        ctrs = model_json["features_info"]["ctrs"]
        ctr_data = model_json["ctr_data"]
        config["scenarios"][tag] = {
            "max_ctr_complexity": 1 if tag == "simple" else 2,
            "n_ctrs": len(ctrs),
            "n_ctr_data_tables": len(ctr_data),
            "ctr_types_in_splits": sorted({c["ctr_type"] for c in ctrs}),
            "ctr_types_in_tables": sorted({json.loads(k)["type"] for k in ctr_data}),
        }
    with open(os.path.join(HERE, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    derive_from_committed()
    # Uncomment to also attempt a best-effort scratch re-run (requires
    # `catboost` installed; writes to `_scratch_regen/`, never overwrites the
    # committed fixtures):
    # regenerate_scratch_copy()
