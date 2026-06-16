#!/usr/bin/env python3
"""OFFLINE ranking-corpus + per-loss/per-metric fixture generator (Phase 6.3,
LOSS-04 / LOSS-05).

RUN-ONCE / COMMIT discipline (D-08): this is a BUILD-TIME tool that runs OUTSIDE
CI on a machine with catboost==1.2.10 importable, and writes committed frozen
`.npy` fixtures. CI only READS the committed artifacts; this generator is NEVER
invoked from CI. Re-run by hand to (re)generate a fixture, then COMMIT.

It owns the SHARED ranking corpus every Wave A-D oracle consumes: a small
deterministic ranking dataset with a contiguous/unique `group_id` column of
VARIED group sizes, a `subgroup_id` column, an explicit `pairs` set (winner/loser
within groups), a numeric feature matrix, and a graded relevance target. The
corpus inputs are written once under `ranking_corpus/inputs/`; each `--loss` /
`--metric` invocation trains (or evaluates) ONE pinned scenario and emits its
per-stage arrays under `ranking_corpus/<name>/`.

This task (Plan 06.3-01) delivers the GENERATOR + corpus spec + README. The
specific per-loss `.npy` fixtures are produced by Plans 02-05 invoking this
generator with their loss/metric name (the deterministic Python-reachable losses;
the randomized YetiRank/StochasticRank ground truth is a separate INSTRUMENTED
C++ generator in Wave C, NOT this file).

PINNED PARAMS (uniform across the whole corpus, mirroring prior phases):
    thread_count=1, depth=2, iterations=5, leaf_estimation_iterations=1,
    boosting_type='Plain' (the *Pairwise variants force Plain -- pinning Plain
    for the whole corpus keeps fixtures uniform), bootstrap_type='No',
    random_strength=0, learning_rate fixed, random_seed fixed.

Determinism: thread_count=1 + fixed random_seed make the summation order
reproducible; the exact params + seed + corpus shape are recorded in this file's
constants and in `fixtures/ranking_corpus/README.md` so later waves regenerate
identically.

Run (OFFLINE, after ensuring `.venv` has catboost==1.2.10):
    .venv/bin/python gen_ranking_fixtures.py --inputs            # write corpus inputs
    .venv/bin/python gen_ranking_fixtures.py --loss QueryRMSE    # one loss fixture
    .venv/bin/python gen_ranking_fixtures.py --metric NDCG       # one metric fixture
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

GENERATOR_DIR = Path(__file__).resolve().parent
FIXTURES = GENERATOR_DIR.parent / "fixtures"
RANKING_CORPUS = FIXTURES / "ranking_corpus"
INPUTS = RANKING_CORPUS / "inputs"

CATBOOST_VERSION = "1.2.10"

# ---------------------------------------------------------------------------
# FROZEN CORPUS SHAPE (Wave-0 spec, VALIDATION.md). Five contiguous groups of
# VARIED size (3, 2, 4, 1, 2) over 12 objects; a subgroup_id column; an explicit
# within-group pair set. group_id is contiguous AND unique (each id appears in
# exactly one contiguous run) -- the shape build_query_info / upstream
# GroupSamples assert.
# ---------------------------------------------------------------------------
GROUP_SIZES = [3, 2, 4, 1, 2]                    # sum == 12 == N_ROWS
N_ROWS = sum(GROUP_SIZES)
RANDOM_SEED = 20260617
LEARNING_RATE = 0.3
DEPTH = 2
ITERATIONS = 5
LEAF_ESTIMATION_ITERATIONS = 1

# Per-object group ids (contiguous, unique per group): [0,0,0, 1,1, 2,2,2,2, 3, 4,4].
GROUP_ID = np.concatenate(
    [np.full(size, g, dtype=np.uint64) for g, size in enumerate(GROUP_SIZES)]
)
# Per-object subgroup ids: enumerate within each group (group-local 0..size).
SUBGROUP_ID = np.concatenate(
    [np.arange(size, dtype=np.uint64) for size in GROUP_SIZES]
)

# Explicit within-group pairs (winner_id, loser_id) by GLOBAL object index. Each
# pair's endpoints fall in the SAME group (a cross-group pair is rejected by the
# Rust builder). Higher-relevance objects win.
#   group 0 = objs [0,1,2]; group 1 = [3,4]; group 2 = [5,6,7,8]; group 3 = [9];
#   group 4 = [10,11].
PAIRS = np.array(
    [
        [0, 2],   # group 0
        [1, 2],   # group 0
        [3, 4],   # group 1
        [5, 8],   # group 2
        [6, 7],   # group 2
        [5, 7],   # group 2
        [10, 11], # group 4 (group 3 is a singleton -> no pairs)
    ],
    dtype=np.uint32,
)

# A small deterministic numeric feature matrix (4 features) + graded relevance
# target in {0,1,2,3}. Generated from a fixed RNG so the corpus is frozen.
def _build_features_and_target() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)
    x = rng.normal(size=(N_ROWS, 4)).astype(np.float64)
    # Graded relevance correlated with the first feature so the model has signal,
    # quantized to {0,1,2,3}.
    raw = x[:, 0] + 0.5 * x[:, 1]
    ranks = np.argsort(np.argsort(raw))
    y = (ranks * 4 // N_ROWS).clip(0, 3).astype(np.float64)
    return x, y


def write_inputs() -> None:
    """Write the frozen corpus inputs (X / y / group_id / subgroup_id / pairs)."""
    INPUTS.mkdir(parents=True, exist_ok=True)
    x, y = _build_features_and_target()
    np.save(INPUTS / "X.npy", x, allow_pickle=False)
    np.save(INPUTS / "y.npy", y, allow_pickle=False)
    np.save(INPUTS / "group_id.npy", GROUP_ID, allow_pickle=False)
    np.save(INPUTS / "subgroup_id.npy", SUBGROUP_ID, allow_pickle=False)
    np.save(INPUTS / "pairs.npy", PAIRS, allow_pickle=False)
    meta = {
        "catboost_version": CATBOOST_VERSION,
        "n_rows": N_ROWS,
        "group_sizes": GROUP_SIZES,
        "random_seed": RANDOM_SEED,
        "pinned_params": {
            "thread_count": 1,
            "depth": DEPTH,
            "iterations": ITERATIONS,
            "leaf_estimation_iterations": LEAF_ESTIMATION_ITERATIONS,
            "boosting_type": "Plain",
            "bootstrap_type": "No",
            "random_strength": 0,
            "learning_rate": LEARNING_RATE,
        },
    }
    (INPUTS / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote corpus inputs under {INPUTS}")


def _load_inputs() -> tuple[np.ndarray, np.ndarray]:
    if not (INPUTS / "X.npy").exists():
        write_inputs()
    x = np.load(INPUTS / "X.npy", allow_pickle=False)
    y = np.load(INPUTS / "y.npy", allow_pickle=False)
    return x, y


def _make_pool(x: np.ndarray, y: np.ndarray):
    """Build a catboost Pool carrying the frozen group/subgroup/pair structure."""
    from catboost import Pool

    return Pool(
        data=x,
        label=y,
        group_id=GROUP_ID.astype(np.int64),
        subgroup_id=SUBGROUP_ID.astype(np.int64),
        pairs=PAIRS.astype(np.int64),
    )


def _pinned_params(loss: str) -> dict:
    """The uniform pinned param set for one ranking loss/objective."""
    return {
        "loss_function": loss,
        "iterations": ITERATIONS,
        "depth": DEPTH,
        "learning_rate": LEARNING_RATE,
        "leaf_estimation_iterations": LEAF_ESTIMATION_ITERATIONS,
        "boosting_type": "Plain",
        "bootstrap_type": "No",
        "random_strength": 0,
        "random_seed": RANDOM_SEED,
        "thread_count": 1,
        "allow_writing_files": False,
    }


def gen_loss(loss: str) -> None:
    """Train ONE pinned ranking-loss scenario; emit per-stage fixtures.

    Emits under `ranking_corpus/<loss>/`:
        model.json        splits + leaf_values (Stage::Splits / Stage::LeafValues)
        staged.npy        per-iteration staged approx (Stage::StagedApprox)
        predictions.npy   final predictions (Stage::Predictions)
        config.json       the exact pinned params (auditable baseline)
    """
    from catboost import CatBoost

    x, y = _load_inputs()
    pool = _make_pool(x, y)
    scenario = RANKING_CORPUS / loss
    scenario.mkdir(parents=True, exist_ok=True)

    params = _pinned_params(loss)
    model = CatBoost(params)
    model.fit(pool)

    model.save_model(str(scenario / "model.json"), format="json")
    staged = [np.asarray(p, dtype=np.float64) for p in model.staged_predict(pool)]
    staged_flat = np.concatenate([s.reshape(-1) for s in staged]) if staged else np.array([])
    np.save(scenario / "staged.npy", staged_flat.astype(np.float64), allow_pickle=False)
    predictions = np.asarray(model.predict(pool), dtype=np.float64)
    np.save(scenario / "predictions.npy", predictions.reshape(-1), allow_pickle=False)
    (scenario / "config.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    print(f"wrote loss fixture under {scenario}")


def gen_metric(metric: str) -> None:
    """Evaluate ONE ranking metric over the corpus; emit the reference value(s).

    Ranking metrics are eval-only (no der). This trains a small fixed model with
    a default ranking objective, then computes `metric` over the corpus via
    catboost's eval_metrics, emitting:
        metric_value.npy   per-iteration metric value(s) (the LOSS-05 reference)
        config.json        the exact pinned params + metric spec
    """
    from catboost import CatBoost

    x, y = _load_inputs()
    pool = _make_pool(x, y)
    scenario = RANKING_CORPUS / metric
    scenario.mkdir(parents=True, exist_ok=True)

    # A fixed pairwise objective so the corpus has a trained model to score the
    # eval-only metric against; the metric value is what the Rust EvalMetric must
    # reproduce <= 1e-5 (D-6.3-05).
    params = _pinned_params("PairLogit")
    model = CatBoost(params)
    model.fit(pool)
    values = model.eval_metrics(pool, [metric])
    arr = np.asarray(values[metric], dtype=np.float64)
    np.save(scenario / "metric_value.npy", arr.reshape(-1), allow_pickle=False)
    cfg = dict(params)
    cfg["eval_metric"] = metric
    (scenario / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"wrote metric fixture under {scenario}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", action="store_true", help="write corpus inputs only")
    parser.add_argument("--loss", type=str, default=None, help="ranking loss name (e.g. QueryRMSE)")
    parser.add_argument("--metric", type=str, default=None, help="ranking metric name (e.g. NDCG)")
    args = parser.parse_args(argv)

    RANKING_CORPUS.mkdir(parents=True, exist_ok=True)

    if args.inputs:
        write_inputs()
    if args.loss is not None:
        gen_loss(args.loss)
    if args.metric is not None:
        gen_metric(args.metric)
    if not (args.inputs or args.loss or args.metric):
        # Default: (re)write the corpus inputs so the frozen shape is materialized.
        write_inputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
