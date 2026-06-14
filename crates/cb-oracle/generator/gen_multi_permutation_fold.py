#!/usr/bin/env python3
"""Generate the multi_permutation_fold AveragingFold draw-order oracle fixture
(ORD-01 / WR-01, Plan 05-15) from a REAL catboost 1.2.10 dump.

RUN-ONCE / COMMIT discipline (D-11/D-12): this is a BUILD-TIME tool run by hand
OUTSIDE CI. It imports `catboost` from the project `.venv`, trains the
`tensor_ctr_e2e` config family at `permutation_count=1` (baseline), `=2`
(learning_folds == 1) and `=4` (learning_folds == 3), and COMMITS catboost's
OBSERVABLE upstream output: the trained-model tree-0 `leaf_weights` — the
AveragingFold partition counts (05-CTR-LEAF-VALUE-RESEARCH.md Open-Q5:
"model.json leaf_weights are the AveragingFold (shuffled) partition counts"). CI
only READS the committed artifacts.

Why leaf_weights are the upstream anchor (not a self-oracle). WR-01 flags that
the "exactly one pre-averaging GenRand at idx==learning_folds" advance count was
NEVER validated against catboost for permutation_count>1. catboost does not
expose the internal AveragingFold->LearnPermutation array through its Python API,
but the tree-0 `leaf_weights` ARE a direct, deterministic function of that
permutation (the partition counts of the 30 objects into the 4 oblivious leaves
under the AveragingFold-permuted online-prefix CTR). A WRONG advance count
produces a DIFFERENT AveragingFold permutation, hence a DIFFERENT partition,
hence DIFFERENT leaf_weights — so asserting cb_train's averaging permutation
reproduces catboost's committed leaf_weights validates the advance count against
catboost itself, which is exactly the WR-01 risk. This is observable upstream
output, not cb-core's own RNG.

The committed dump (`leaf_weights.json` + the full `model_pc{N}.json`) is the
upstream authority the Rust oracle (`multi_permutation_fold_oracle_test.rs`)
checks `cb_train::create_folds` against integer-exact. The Rust oracle ALSO keeps
a self-derived TFastRng64 cross-check as a SECONDARY consistency assertion (never
the authority — a self-oracle bakes in the same advance-count assumption).

Empirically (catboost 1.2.10, tensor_ctr_e2e config family):
    pc=1: tree0 leaf_weights = [6, 0, 7, 17]   (learning_folds == 1)
    pc=2: tree0 leaf_weights = [6, 0, 7, 17]   (learning_folds == 1; same draw stream as pc=1)
    pc=4: tree0 leaf_weights = [6, 0, 10, 14]  (learning_folds == 3; two extra learning shuffles before the pre-averaging draw)
The pc=2 == pc=1 equality is the smoking-gun WR-01 check: with the CORRECT
(idx == learning_folds) guard the averaging shuffle is preceded by zero learning
shuffles at both pc=1 and pc=2 (learning_folds == 1), so the partition matches;
the OLD (first learning shuffle) guard would have diverged pc=2 from upstream.
The pc=4 [6,0,10,14] partition can ONLY be reproduced if the two extra learning
shuffles (folds 1 and 2) consume draws BEFORE the pre-averaging GenRand — i.e.
the advance count is correct.

Run (catboost==1.2.10 must be importable):
    .venv/bin/python crates/cb-oracle/generator/gen_multi_permutation_fold.py
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
from catboost import CatBoost, Pool

GENERATOR_DIR = Path(__file__).resolve().parent
TENSOR_FX = GENERATOR_DIR.parent / "fixtures" / "tensor_ctr_e2e"
# The fixture lives under cb-train's tests/ (the plan's files_modified target).
OUT_DIR = (
    GENERATOR_DIR.parent.parent
    / "cb-train"
    / "tests"
    / "fixtures"
    / "multi_permutation_fold"
)

N = 30
SEED = 0


def main() -> None:
    if not TENSOR_FX.exists():
        raise SystemExit(f"missing tensor_ctr_e2e fixture at {TENSOR_FX}")
    X = np.load(TENSOR_FX / "X_cat.npy")
    y = np.load(TENSOR_FX / "y.npy")
    assert X.shape[0] == N, X.shape

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base_params = dict(
        loss_function="Logloss",
        iterations=5,
        depth=2,
        learning_rate=0.1,
        l2_leaf_reg=3.0,
        random_strength=0,
        bootstrap_type="No",
        boost_from_average=False,
        leaf_estimation_method="Gradient",
        leaf_estimation_iterations=1,
        one_hot_max_size=1,
        max_ctr_complexity=2,
        simple_ctr=["Borders:Prior=0.5"],
        combinations_ctr=["Borders:Prior=0.5"],
        fold_len_multiplier=2.0,
        random_seed=SEED,
        thread_count=1,
        verbose=False,
        boosting_type="Plain",
    )

    leaf_weights_by_pc = {}
    for pc in (1, 2, 4):
        params = dict(base_params, permutation_count=pc)
        model = CatBoost(params)
        pool = Pool(X, label=y, cat_features=[0, 1])
        model.fit(pool)

        td = tempfile.mkdtemp()
        mj = os.path.join(td, "m.json")
        model.save_model(mj, format="json")
        with open(mj) as fh:
            mjson = json.load(fh)
        tree0_lw = [int(round(w)) for w in mjson["oblivious_trees"][0]["leaf_weights"]]
        leaf_weights_by_pc[str(pc)] = {
            "permutation_count": pc,
            "learning_folds": max(1, pc - 1),
            "tree0_leaf_weights": tree0_lw,
        }
        # Commit the full upstream model.json for auditability.
        with open(OUT_DIR / f"model_pc{pc}.json", "w") as fh:
            json.dump(mjson, fh)
        print(f"pc={pc}: tree0_leaf_weights={tree0_lw}")

    with open(OUT_DIR / "leaf_weights.json", "w") as fh:
        json.dump(leaf_weights_by_pc, fh, indent=2, sort_keys=True)

    config = {
        "catboost_version": "1.2.10",
        "scenario": "multi_permutation_fold",
        "requirement": "ORD-01",
        "n_rows": N,
        "seed": SEED,
        "note": (
            "REAL catboost 1.2.10 AveragingFold draw-order anchor for "
            "permutation_count>=2 (WR-01). leaf_weights.json records catboost's "
            "trained-model tree-0 leaf_weights (the AveragingFold partition counts) "
            "for pc=1,2,4 of the tensor_ctr_e2e config family. The Rust oracle "
            "asserts the partition cb_train::create_folds's AveragingFold "
            "permutation produces over the online-prefix CTR column equals these "
            "committed catboost leaf_weights integer-exact. A wrong pre-averaging "
            "advance count would yield a different partition and FAIL the check. "
            "model_pc{1,2,4}.json are the full upstream dumps for audit. Generated "
            "OFFLINE; NEVER run in CI (D-12)."
        ),
        "params": dict(base_params),
        "leaf_weights": leaf_weights_by_pc,
    }
    with open(OUT_DIR / "config.json", "w") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
    print(f"wrote fixture to {OUT_DIR}")


if __name__ == "__main__":
    main()
