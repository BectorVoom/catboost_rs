#!/usr/bin/env python3
"""Generate the SIMPLEST-possible Depthwise non-symmetric SPLITS preflight fixture
(Phase 06.6 plan 04, Task 1 — RESEARCH §"Open Questions (RESOLVED)" Q1).

RUN-ONCE / COMMIT discipline (mirrors gen_monotone_fixtures.py): a BUILD-TIME tool
run OUTSIDE CI that writes committed frozen fixtures. CI only READS them; this
generator is NEVER invoked from CI. Re-run by hand only to regenerate, then COMMIT.

PURPOSE — EARLY draw-stream divergence detection (RESEARCH Open Question 1).
Before the leaf-wise grower is written (06.6-04 Task 2), this fixture isolates the
single empirical question that decides whether our grower reproduces upstream's
non-symmetric SPLITS: does the Depthwise level-order draw stream match catboost
1.2.10? To isolate ONLY that question, every confounding source of variation is
pinned OFF:

    * grow_policy='Depthwise'         — the policy under test
    * random_strength=0               — no per-candidate score perturbation (no RNG)
    * bootstrap_type='No'             — no per-object sampling (no RNG)
    * thread_count=1                  — deterministic, single-host reduction order
    * NO categorical features         — no CTR machinery, no CTR draw stream
    * max_depth small (2)             — the SMALLEST non-trivial level-order tree
    * iterations small (2)            — minimal boosting feedback
    * boost_from_average=False        — constant bias path
    * leaf_estimation_iterations=1    — one Newton/Gradient step
    * random_seed pinned (42)

The result is the simplest non-symmetric model whose SPLITS can possibly differ
from ours ONLY through the level-order expansion / candidate-enumeration draw
stream — exactly RESEARCH Open Question 1.

ESCALATE-DON'T-WEAKEN (D-6.6-11): if, once the grower lands (Task 2), the
`depthwise_simplest_splits` preflight diverges, the draw stream differs from
upstream. The resolution is to ESCALATE to the persistent instrumented trainer
(`/tmp/cb_build313` + clang-18) to capture the exact upstream draw order — NEVER
loosen the tolerance, NEVER `#[ignore]` the test, NEVER fabricate splits.

Outputs (layout matches the existing non_symmetric fixtures so the cb-model
`non_symmetric_oracle_test.rs` harness reuses `fixture()` / `load_model_json` /
`compare_stage`):
    save_model(format=json)  -> model.json    (the nested "trees" + split borders)
    save_model(format=cbm)   -> model.cbm      (the flat non-symmetric .cbm)
    X.npy                    -> the float feature matrix (no cat features)
    staged_predict()         -> staged.npy     (n_iterations x n_rows, f64)
    predict()                -> predictions.npy
    meta.json                -> the pinned params (provenance)

Run:
    .venv/bin/python crates/cb-oracle/generator/gen_depthwise_simplest_fixture.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from catboost import CatBoost, Pool

# --- Pinned, isolating parameters (see module docstring) --------------------
SEED = 42
N_OBJECTS = 48
N_FEATURES = 2
MAX_DEPTH = 2
ITERATIONS = 2

PARAMS = {
    "iterations": ITERATIONS,
    "learning_rate": 0.3,
    "loss_function": "RMSE",
    "grow_policy": "Depthwise",
    "max_depth": MAX_DEPTH,
    "random_strength": 0,
    "bootstrap_type": "No",
    "thread_count": 1,
    "random_seed": SEED,
    "feature_border_type": "GreedyLogSum",
    "border_count": 32,
    "leaf_estimation_iterations": 1,
    "boost_from_average": False,
    "allow_writing_files": False,
}


def main() -> None:
    out_dir = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "non_symmetric"
        / "depthwise_simplest"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    # Pure float features, no categoricals -> no CTR machinery.
    X = rng.random((N_OBJECTS, N_FEATURES)).astype(np.float64)
    # A smooth target with a mild interaction so Depthwise actually splits.
    y = (X[:, 0] * 2.0 - X[:, 1] + 0.3 * X[:, 0] * X[:, 1]).astype(np.float64)

    pool = Pool(data=X, label=y)
    model = CatBoost(PARAMS)
    model.fit(pool)

    model.save_model(str(out_dir / "model.json"), format="json")
    model.save_model(str(out_dir / "model.cbm"), format="cbm")

    with open(out_dir / "model.json") as f:
        mj = json.load(f)
    assert "trees" in mj and len(mj["trees"]) == ITERATIONS, (
        "fixture must be a non-symmetric ('trees') model with the pinned tree count"
    )

    np.save(out_dir / "X.npy", X)

    staged = np.array(
        list(model.staged_predict(pool, prediction_type="RawFormulaVal")),
        dtype=np.float64,
    )
    np.save(out_dir / "staged.npy", staged)

    preds = model.predict(pool, prediction_type="RawFormulaVal").astype(np.float64)
    np.save(out_dir / "predictions.npy", preds)

    meta = {
        "catboost_version": __import__("catboost").__version__,
        "grow_policy": "Depthwise",
        "params": PARAMS,
        "n_objects": N_OBJECTS,
        "n_features": N_FEATURES,
        "seed": SEED,
        "note": (
            "offline-generated from .venv catboost 1.2.10; SIMPLEST-Depthwise "
            "SPLITS preflight (FEAT-06, RESEARCH Open Question 1). All confounds "
            "(random_strength, bootstrap, CTR, threading, boost_from_average) "
            "pinned OFF to isolate the level-order draw stream. ESCALATE (D-6.6-11) "
            "to /tmp/cb_build313 instrumented trainer on divergence; never weaken."
        ),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote simplest-Depthwise fixture to {out_dir}")
    print(f"  trees={len(mj['trees'])} depth={MAX_DEPTH} n_obj={N_OBJECTS}")


if __name__ == "__main__":
    main()
