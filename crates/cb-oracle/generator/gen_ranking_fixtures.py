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


# StochasticRank requires an EXPLICIT target metric in its loss spec
# (`StochasticRank:metric=<DCG|NDCG|...>`). The prior corpus config did NOT pin
# one, so no model.json could be trained. We pin DCG to match the Rust
# `StochasticRankMetric::Dcg` arm the end-to-end oracle drives (the DCG arm avoids
# the per-iteration ideal-DCG renormalization NDCG layers on top, keeping the
# trainer-level per-group noise gate the sole RNG variable). The full loss spec
# string upstream parses is `StochasticRank:metric=DCG`.
STOCHASTIC_RANK_METRIC = "DCG"


def _loss_function_spec(loss: str) -> str:
    """The exact `loss_function` string upstream parses for `loss`.

    StochasticRank needs its target metric folded into the spec; every other
    ranking loss is its bare name.
    """
    if loss == "StochasticRank":
        return f"StochasticRank:metric={STOCHASTIC_RANK_METRIC}"
    return loss


def _pinned_params(loss: str) -> dict:
    """The uniform pinned param set for one ranking loss/objective."""
    return {
        "loss_function": _loss_function_spec(loss),
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
    # The randomized losses (yetirank / stochasticrank) live under a LOWERCASE
    # scenario dir (their model.json is the instrumented-trainer fixture the Rust
    # oracle reads at `ranking_corpus/stochasticrank/`). The deterministic losses
    # keep their PascalCase name. Honor the existing on-disk convention.
    scenario_name = "stochasticrank" if loss == "StochasticRank" else loss
    scenario = RANKING_CORPUS / scenario_name
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


# ---------------------------------------------------------------------------
# Per-metric eval-only fixtures (LOSS-05, Plan 06.3-05).
#
# Ranking metrics are eval-only, so the cleanest Python-reachable ground truth is
# `catboost.utils.eval_metric(label, approx, metric, group_id=...)` over a FIXED,
# KNOWN approx vector (NOT a trained-model prediction — the Rust trainer need not
# reproduce a PairLogit model to gate the metric formula). The frozen `approx` +
# `group_id` + `target` are shared by every metric scenario; each scenario freezes
# its upstream scalar value(s). The Rust `EvalMetric::eval_grouped` is fed the same
# approx/target/group_id and compared ≤1e-5.
#
# group_id / target / approx are stored as float64 so the Rust oracle can load them
# all via `cb_oracle::load_f64_vec` (the only npy loader in cb-oracle).
# ---------------------------------------------------------------------------
METRICS_DIR = RANKING_CORPUS / "ranking_metrics"
METRIC_APPROX_SEED = 42

# ---------------------------------------------------------------------------
# ORCH-04-S2 FLAT calc_metrics fixtures (standalone eval_metric surface).
# Metrics on FIXED predictions have no training/quantization nondeterminism, so
# these are the cleanest possible oracle. A SINGLE shared (label, approx) pair is
# reused for RMSE, Logloss, and MSLE simultaneously; this is valid ONLY because
# `label` is pinned to {0, 1} (upstream Logloss requires target in [0,1]; MSLE's
# `1+label>0` holds trivially) and `approx > -1` (the MSLE `1+approx>0`
# log-domain guard, cb-train metrics.rs:326-331). A positive `weight` vector
# yields the weighted-RMSE case (`catboost.utils.eval_metric` accepts `weight=`
# in 1.2.10 -- confirmed at gen time).
CALC_METRICS_DIR = FIXTURES / "calc_metrics"
CALC_METRICS_SEED = 20260718
N_CALC = 16

# The metric scenarios: (scenario_name, catboost_metric_string). Defaults +
# explicit top=2 cases exercise the nth_element / tie path; QueryAUC uses
# type=Ranking for the graded-relevance corpus (the singleclass Classic default
# requires target in [0,1]) plus a Classic case on a binarized target.
METRIC_SCENARIOS = [
    ("ndcg", "NDCG"),
    ("dcg", "DCG"),
    ("map", "MAP"),
    ("mrr", "MRR"),
    ("err", "ERR"),
    ("pfound", "PFound"),
    ("precision_at", "PrecisionAt"),
    ("recall_at", "RecallAt"),
    ("queryauc_ranking", "QueryAUC:type=Ranking"),
    ("ndcg_top2", "NDCG:top=2"),
    ("dcg_top2", "DCG:top=2"),
    ("map_top2", "MAP:top=2"),
    ("mrr_top2", "MRR:top=2"),
    ("err_top2", "ERR:top=2"),
    ("pfound_top2", "PFound:top=2"),
    ("precision_at_top2", "PrecisionAt:top=2"),
    ("recall_at_top2", "RecallAt:top=2"),
]


def _metric_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared (group_id, target, approx, binary_target) for the metric fixtures."""
    _, y = _load_inputs()
    group_id = GROUP_ID.astype(np.float64)
    target = y.astype(np.float64)
    rng = np.random.default_rng(METRIC_APPROX_SEED)
    approx = rng.normal(size=N_ROWS).astype(np.float64)
    # A binarized target (relevance > 1.5 -> 1) for the Classic QueryAUC case,
    # which requires target in [0, 1].
    binary_target = (target > 1.5).astype(np.float64)
    return group_id, target, approx, binary_target


def gen_metrics_eval() -> None:
    """Freeze the per-metric eval-only fixtures over a FIXED approx (LOSS-05)."""
    from catboost.utils import eval_metric

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    group_id, target, approx, binary_target = _metric_inputs()

    # Shared inputs (float64 so cb_oracle::load_f64_vec can read them).
    np.save(METRICS_DIR / "group_id.npy", group_id, allow_pickle=False)
    np.save(METRICS_DIR / "target.npy", target, allow_pickle=False)
    np.save(METRICS_DIR / "approx.npy", approx, allow_pickle=False)
    np.save(METRICS_DIR / "binary_target.npy", binary_target, allow_pickle=False)

    group_id_int = GROUP_ID.astype(np.int64)
    summary = {"catboost_version": CATBOOST_VERSION, "approx_seed": METRIC_APPROX_SEED, "scenarios": {}}
    for name, metric in METRIC_SCENARIOS:
        value = eval_metric(target, approx, metric, group_id=group_id_int)
        arr = np.asarray(value, dtype=np.float64)
        np.save(METRICS_DIR / f"{name}.npy", arr.reshape(-1), allow_pickle=False)
        summary["scenarios"][name] = {"metric": metric, "value": [float(v) for v in arr.reshape(-1)]}

    # The Classic QueryAUC case uses the binarized target (target in [0,1]).
    qauc_classic = eval_metric(binary_target, approx, "QueryAUC:type=Classic", group_id=group_id_int)
    arr = np.asarray(qauc_classic, dtype=np.float64).reshape(-1)
    np.save(METRICS_DIR / "queryauc_classic.npy", arr, allow_pickle=False)
    summary["scenarios"]["queryauc_classic"] = {
        "metric": "QueryAUC:type=Classic (binary_target)",
        "value": [float(v) for v in arr],
    }

    (METRICS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {len(summary['scenarios'])} metric fixtures under {METRICS_DIR}")


def gen_calc_metrics_flat() -> None:
    """Freeze the FLAT calc_metrics fixtures over a FIXED (label, approx) pair
    (ORCH-04-S2). RMSE/Logloss/MSLE share the one pair; a positive weight vector
    yields the weighted-RMSE case."""
    from catboost.utils import eval_metric

    CALC_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(CALC_METRICS_SEED)
    # label in {0,1}: satisfies RMSE + Logloss (target in [0,1]) + MSLE (1+label>0).
    label = rng.integers(0, 2, size=N_CALC).astype(np.float64)
    # approx > -1: MSLE log-domain guard (1+approx>0); RMSE/Logloss accept any raw.
    approx = rng.uniform(-0.5, 2.0, size=N_CALC).astype(np.float64)
    # strictly-positive per-object weights for the weighted-RMSE case.
    weight = rng.uniform(0.5, 2.0, size=N_CALC).astype(np.float64)

    # Shared inputs (float64 so cb_oracle::load_f64_vec can read them).
    np.save(CALC_METRICS_DIR / "label.npy", label, allow_pickle=False)
    np.save(CALC_METRICS_DIR / "approx.npy", approx, allow_pickle=False)
    np.save(CALC_METRICS_DIR / "weight.npy", weight, allow_pickle=False)

    summary = {
        "catboost_version": CATBOOST_VERSION,
        "seed": CALC_METRICS_SEED,
        "n": N_CALC,
        "weight_supported": True,
        "scenarios": {},
    }

    def freeze(name: str, metric: str, **kw) -> None:
        value = eval_metric(label, approx, metric, **kw)
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name}: non-finite metric value {arr!r}")
        np.save(CALC_METRICS_DIR / f"{name}.npy", arr, allow_pickle=False)
        summary["scenarios"][name] = {
            "metric": metric,
            "weighted": bool(kw),
            "value": [float(v) for v in arr],
        }

    def freeze_weighted(name: str, metric: str) -> None:
        """Freeze a weighted scenario ONLY if `catboost.utils.eval_metric`
        actually accepts `weight=` for `metric` in this catboost version. A
        `weight=`-unsupported metric must NOT silently emit a bogus fixture
        (EMT-6): skip it and record the skip in summary.json."""
        try:
            freeze(name, metric, weight=weight)
        except Exception as exc:  # noqa: BLE001 — record + skip, never emit bogus
            summary["scenarios"][name] = {
                "metric": metric,
                "weighted": True,
                "skipped": f"weight= unsupported: {type(exc).__name__}: {exc}",
            }
            print(f"skip {name}: weight= unsupported for {metric}: {exc}")

    freeze("rmse", "RMSE")
    freeze("logloss", "Logloss")
    freeze("msle", "MSLE")
    freeze("rmse_weighted", "RMSE", weight=weight)

    # EM-05 — flat Min-optimized metrics (EMT-6). The frozen `{0,1}` label
    # carries zero-target rows, so `mape` is the R1 divisor arbiter.
    freeze("mae", "MAE")
    freeze("mape", "MAPE")
    freeze("quantile_default", "Quantile")  # alpha default 0.5
    freeze("quantile_a90", "Quantile:alpha=0.9")  # R2 asymmetric alpha
    # Weighted variants — verified `weight=`-supported for MAE/Quantile in
    # catboost 1.2.10, but freeze via try/skip so a version that rejects
    # `weight=` does not emit a bogus fixture.
    freeze_weighted("mae_weighted", "MAE")
    freeze_weighted("quantile_default_weighted", "Quantile")

    (CALC_METRICS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(summary['scenarios'])} flat calc_metrics fixtures under {CALC_METRICS_DIR}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", action="store_true", help="write corpus inputs only")
    parser.add_argument("--loss", type=str, default=None, help="ranking loss name (e.g. QueryRMSE)")
    parser.add_argument("--metric", type=str, default=None, help="ranking metric name (e.g. NDCG)")
    parser.add_argument(
        "--metrics-eval",
        action="store_true",
        help="freeze all eval-only ranking metric fixtures (LOSS-05, Plan 06.3-05)",
    )
    parser.add_argument(
        "--calc-metrics",
        action="store_true",
        help="freeze the FLAT calc_metrics fixtures (ORCH-04-S2: RMSE/Logloss/MSLE + weighted)",
    )
    args = parser.parse_args(argv)

    RANKING_CORPUS.mkdir(parents=True, exist_ok=True)

    if args.inputs:
        write_inputs()
    if args.loss is not None:
        gen_loss(args.loss)
    if args.metric is not None:
        gen_metric(args.metric)
    if args.metrics_eval:
        gen_metrics_eval()
    if args.calc_metrics:
        gen_calc_metrics_flat()
    if not (
        args.inputs
        or args.loss
        or args.metric
        or args.metrics_eval
        or args.calc_metrics
    ):
        # Default: (re)write the corpus inputs so the frozen shape is materialized.
        write_inputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
