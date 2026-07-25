#!/usr/bin/env python3
"""OFFLINE fixed-fold cross-validation oracle fixture generator (Phase 20,
ORCH-01 / TASK-01, supports ORCH-01-S5).

RUN-ONCE / COMMIT discipline (D-08): this is a BUILD-TIME tool that runs OUTSIDE
CI on a machine with catboost==1.2.10 importable, and writes committed frozen
`.npy` / `.json` fixtures. CI (the Rust `cv_oracle_test`) only READS the committed
artifacts; this generator is NEVER invoked from CI. Re-run by hand to
(re)generate the fixture, then COMMIT.

It emits, under `fixtures/cv/`, a fixed numeric-regression dataset, an explicit
fixed KFold(n_splits=3, shuffle=False) fold assignment, the pinned param set, and
the upstream `catboost.cv` per-iteration aggregate columns
(`test-RMSE-mean/std`, `train-RMSE-mean/std`) as float64 `.npy` vectors of length
`iterations`, so the Rust facade `cv()` can be oracle-compared ≤1e-5.

CRITICAL fixture-generation requirement (SPEC §1.1, PLAN TASK-01 — MANDATORY, not
optional tuning): the pinned `params` dict MUST explicitly set
`"boost_from_average": True`, `"bootstrap_type": "No"` AND `"random_strength": 0`.
`catboost.cv()`'s raw, dict-based native call does NOT apply the
`CatBoostRegressor` estimator class's Python-side default-injection, so
`boost_from_average` is silently `False` (bias forced to 0) and the bootstrap
default differs, at the raw `_cv()` layer. This project's `CatBoostBuilder`
defaults `boost_from_average: true`, `bootstrap_type: EBootstrapType::No`, and —
crucially — `random_strength: 0.0` (`crates/catboost-rs/src/builder.rs:107,108,110`),
whereas catboost's own default is `random_strength=1.0`. Omitting any of these keys
makes the fixture NOT reproduce what the Rust `cv()` under test actually trains,
independent of whether the Rust implementation is correct. Explicitly setting all
three collapses the aggregate divergence to ~1e-8 across every fold (all trees
bit-identical).

`random_strength` was the default that TASK-01's border/bias-only parity
self-check could NOT catch: `random_strength` perturbs split SCORES, not the
feature borders or the scale/bias, so the per-fold `get_borders()` +
`get_scale_and_bias()` self-check reported 0 mismatches even while catboost's
`random_strength=1.0` (vs the facade's `0.0`) was silently diverging the chosen
splits and hence the per-iteration RMSE curve. It is now pinned to match the
facade default (`builder.rs:107`).

std convention (SPEC §9 Q2, RESOLVED): `catboost.cv`'s `-std` columns are SAMPLE
std (ddof=1, divide by F-1). This is asserted below against
`numpy.std(per_fold_curve, ddof=1)` and confirmed different from ddof=0.

Systematic parity self-check (PLAN TASK-01): before freezing, for EACH fold this
generator (a) gets that fold's ACTUAL trained model from
`catboost.cv(..., return_models=True)` and (b) fits a standalone
`CatBoostRegressor(**params).fit(Pool(X[train], y[train]))` on the identical row
subset, then asserts `get_borders()` (per feature) AND `get_scale_and_bias()` are
IDENTICAL between the two for every fold. Any mismatch is a still-undiscovered
`CatBoostBuilder`-default that upstream's raw `cv()` also fails to inject, and
must be root-caused and folded into the pinned `params` dict BEFORE freezing.

Run (OFFLINE, after ensuring the venv has catboost==1.2.10, numpy<2, scikit-learn):
    uv venv --python 3.12 && uv pip install catboost==1.2.10 'numpy<2' scikit-learn
    .venv/bin/python crates/cb-oracle/generator/gen_cv_fixtures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

import catboost
from catboost import CatBoostRegressor, Pool

GENERATOR_DIR = Path(__file__).resolve().parent
FIXTURES = GENERATOR_DIR.parent / "fixtures"
CV_DIR = FIXTURES / "cv"

CATBOOST_VERSION = "1.2.10"

# --------------------------------------------------------------------------- #
# FROZEN DATASET SHAPE. A deterministic numeric-regression dataset: 30 objects,
# 4 float64 features, a continuous float64 target with real signal so RMSE has a
# non-trivial per-iteration descent. Fixed RNG => frozen corpus.
# --------------------------------------------------------------------------- #
N_ROWS = 30
N_FEATURES = 4
DATA_SEED = 20260720
N_SPLITS = 3

# PINNED PARAMS (committed verbatim to params.json; the Rust oracle builds its
# CatBoostBuilder from EXACTLY these keys). boost_from_average + bootstrap_type +
# random_strength are MANDATORY (see module docstring / SPEC §1.1). random_strength
# is pinned to 0 to match the facade default (builder.rs:107); catboost's own raw
# default is 1.0, which perturbs split scores and diverges the RMSE curve.
PARAMS = {
    "iterations": 10,
    "learning_rate": 0.1,
    "depth": 4,
    "loss_function": "RMSE",
    "border_count": 128,
    "boost_from_average": True,
    "bootstrap_type": "No",
    "random_strength": 0,
}

# Default metric derived from loss_function="RMSE" is RMSE (SPEC §9 Q3).
DEFAULT_METRIC = "RMSE"
# Sample std, ddof=1 (SPEC §9 Q2, RESOLVED).
STD_DDOF = 1


def _build_dataset() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(DATA_SEED)
    x = rng.normal(size=(N_ROWS, N_FEATURES)).astype(np.float64)
    # A target with real signal (linear combo + mild nonlinearity + noise).
    y = (
        2.0 * x[:, 0]
        - 1.5 * x[:, 1]
        + 0.75 * x[:, 2] * x[:, 3]
        + 0.3 * rng.normal(size=N_ROWS)
    ).astype(np.float64)
    return x, y


def _kfold_test_lists(x: np.ndarray) -> list[list[int]]:
    """KFold(n_splits=3, shuffle=False) TEST-row index lists (contiguous blocks)."""
    kf = KFold(n_splits=N_SPLITS, shuffle=False)
    return [test_idx.tolist() for _, test_idx in kf.split(x)]


def _cv_params_call() -> dict:
    """Params passed to the training calls. Identical to the committed PARAMS
    plus non-numeric hygiene keys (allow_writing_files/thread hygiene) that do
    NOT alter the trained model's borders/bias/curve; kept out of params.json."""
    p = dict(PARAMS)
    p["allow_writing_files"] = False
    return p


def _parity_self_check(
    x: np.ndarray,
    y: np.ndarray,
    fold_test_lists: list[list[int]],
    cv_models: list,
) -> list[dict]:
    """For each fold assert the cv fold model and a standalone regressor fit on the
    identical train subset share IDENTICAL borders + scale/bias. Returns per-fold
    records for summary.json; raises AssertionError on any mismatch."""
    params = _cv_params_call()
    all_idx = np.arange(len(x))
    records = []
    for k, test_idx in enumerate(fold_test_lists):
        test_set = set(test_idx)
        train_idx = np.array([i for i in all_idx if i not in test_set], dtype=int)

        standalone = CatBoostRegressor(**params)
        standalone.fit(Pool(x[train_idx], y[train_idx]), verbose=False)

        cv_model = cv_models[k]

        cv_borders = cv_model.get_borders()
        st_borders = standalone.get_borders()

        # Compare borders per feature (dict {feature_idx: [borders...]}).
        border_keys_match = sorted(cv_borders.keys()) == sorted(st_borders.keys())
        borders_match = border_keys_match and all(
            np.array_equal(
                np.asarray(cv_borders[fk], dtype=np.float64),
                np.asarray(st_borders[fk], dtype=np.float64),
            )
            for fk in cv_borders
        )

        cv_scale, cv_bias = cv_model.get_scale_and_bias()
        st_scale, st_bias = standalone.get_scale_and_bias()
        scale_bias_match = (
            np.array_equal(np.atleast_1d(cv_scale), np.atleast_1d(st_scale))
            and np.array_equal(np.atleast_1d(cv_bias), np.atleast_1d(st_bias))
        )

        assert borders_match, (
            f"[PARITY] fold {k}: get_borders() differ between cv fold model and "
            f"standalone regressor on identical train subset — a hidden "
            f"CatBoostBuilder default is not injected by raw cv(); root-cause and "
            f"pin it in PARAMS before freezing. cv={cv_borders} standalone={st_borders}"
        )
        assert scale_bias_match, (
            f"[PARITY] fold {k}: get_scale_and_bias() differ — cv=({cv_scale},{cv_bias}) "
            f"standalone=({st_scale},{st_bias}); root-cause and pin before freezing."
        )

        records.append(
            {
                "fold": k,
                "train_size": int(len(train_idx)),
                "test_size": int(len(test_idx)),
                "borders_identical": bool(borders_match),
                "scale_bias_identical": bool(scale_bias_match),
                "scale": float(np.atleast_1d(cv_scale)[0]),
                "bias": float(np.atleast_1d(cv_bias)[0]),
            }
        )
    return records


def _reconstruct_per_fold_test_curves(
    x: np.ndarray,
    y: np.ndarray,
    fold_test_lists: list[list[int]],
    cv_models: list,
) -> np.ndarray:
    """Per-fold, per-iteration TEST RMSE curve (shape [F, iterations]) recovered
    from each cv fold model via eval_metrics on that fold's held-out test set.
    Used solely to CONFIRM the ddof convention of the reported -std column."""
    all_idx = np.arange(len(x))
    curves = []
    for k, test_idx in enumerate(fold_test_lists):
        test_idx = np.asarray(test_idx, dtype=int)
        test_pool = Pool(x[test_idx], y[test_idx])
        evals = cv_models[k].eval_metrics(test_pool, [DEFAULT_METRIC])
        curves.append(np.asarray(evals[DEFAULT_METRIC], dtype=np.float64))
    return np.vstack(curves)


def main() -> None:
    CV_DIR.mkdir(parents=True, exist_ok=True)
    x, y = _build_dataset()
    fold_test_lists = _kfold_test_lists(x)

    # Explicit (train, test) folds for catboost.cv: pass the sklearn splitter
    # object directly (KFold(n_splits=3, shuffle=False)). folds= overrides
    # fold_count/shuffle (upstream raises if combined — SPEC §8), so pass NEITHER.
    kf = KFold(n_splits=N_SPLITS, shuffle=False)

    pool = Pool(x, y)
    params = _cv_params_call()

    cv_out = catboost.cv(
        pool,
        params=params,
        folds=kf,
        as_pandas=True,
        return_models=True,
        verbose=False,
    )
    # return_models=True => (results_df, models)
    df, cv_models = cv_out
    assert len(cv_models) == N_SPLITS, f"expected {N_SPLITS} fold models, got {len(cv_models)}"

    iterations = int(PARAMS["iterations"])

    def col(name: str) -> np.ndarray:
        v = np.asarray(df[name].to_numpy(), dtype=np.float64)
        assert v.shape == (iterations,), f"{name} shape {v.shape} != ({iterations},)"
        assert np.all(np.isfinite(v)), f"{name} has non-finite values: {v}"
        return v

    test_mean = col("test-RMSE-mean")
    test_std = col("test-RMSE-std")
    train_mean = col("train-RMSE-mean")
    train_std = col("train-RMSE-std")

    # ---- MANDATORY per-fold parity self-check (root-cause any mismatch) ----
    parity_records = _parity_self_check(x, y, fold_test_lists, cv_models)

    # ---- ddof confirmation (SPEC §9 Q2): -std is SAMPLE std (ddof=1) ----
    per_fold_test = _reconstruct_per_fold_test_curves(x, y, fold_test_lists, cv_models)
    recon_mean = per_fold_test.mean(axis=0)
    recon_std_ddof1 = per_fold_test.std(axis=0, ddof=1)
    recon_std_ddof0 = per_fold_test.std(axis=0, ddof=0)

    mean_recon_max_abs = float(np.max(np.abs(recon_mean - test_mean)))
    std_ddof1_max_abs = float(np.max(np.abs(recon_std_ddof1 - test_std)))
    std_ddof0_max_abs = float(np.max(np.abs(recon_std_ddof0 - test_std)))

    # Reconstructed mean/ddof1-std must match the reported columns; ddof0 must NOT.
    assert mean_recon_max_abs <= 1e-6, (
        f"reconstructed per-fold mean disagrees with reported test-RMSE-mean: "
        f"max|diff|={mean_recon_max_abs}"
    )
    assert std_ddof1_max_abs <= 1e-6, (
        f"reported test-RMSE-std is NOT sample std (ddof=1): "
        f"max|diff(ddof=1)|={std_ddof1_max_abs}"
    )
    assert std_ddof0_max_abs > 1e-6, (
        f"ddof=0 unexpectedly matches reported -std (max|diff|={std_ddof0_max_abs}); "
        f"ddof convention ambiguous — investigate before freezing"
    )

    # --------------------------- write fixtures --------------------------- #
    np.save(CV_DIR / "X.npy", x, allow_pickle=False)
    np.save(CV_DIR / "y.npy", y, allow_pickle=False)
    np.save(CV_DIR / "test_rmse_mean.npy", test_mean, allow_pickle=False)
    np.save(CV_DIR / "test_rmse_std.npy", test_std, allow_pickle=False)
    np.save(CV_DIR / "train_rmse_mean.npy", train_mean, allow_pickle=False)
    np.save(CV_DIR / "train_rmse_std.npy", train_std, allow_pickle=False)

    (CV_DIR / "folds.json").write_text(
        json.dumps([list(map(int, f)) for f in fold_test_lists], indent=2),
        encoding="utf-8",
    )
    (CV_DIR / "params.json").write_text(
        json.dumps(PARAMS, indent=2), encoding="utf-8"
    )

    summary = {
        "catboost_version": CATBOOST_VERSION,
        "dataset": {
            "n_rows": N_ROWS,
            "n_features": N_FEATURES,
            "data_seed": DATA_SEED,
            "target": "2*x0 - 1.5*x1 + 0.75*x2*x3 + 0.3*N(0,1)",
        },
        "params": PARAMS,
        "folds_type": "KFold(n_splits=3, shuffle=False)",
        "folds_test_indices": [list(map(int, f)) for f in fold_test_lists],
        "default_metric": DEFAULT_METRIC,
        "std_ddof": STD_DDOF,
        "std_convention": "sample std (ddof=1, divide by F-1)",
        "parity_self_check": {
            "mismatches": 0,
            "per_fold": parity_records,
            "note": (
                "For every fold, catboost.cv(return_models=True) fold model and a "
                "standalone CatBoostRegressor(**params).fit(Pool(X[train],y[train])) "
                "on the identical train subset have IDENTICAL get_borders() and "
                "get_scale_and_bias(). Zero mismatches => the pinned params fully "
                "reproduce upstream cv's per-fold training."
            ),
        },
        "ddof_confirmation": {
            "reconstructed_mean_max_abs_diff": mean_recon_max_abs,
            "std_ddof1_max_abs_diff": std_ddof1_max_abs,
            "std_ddof0_max_abs_diff": std_ddof0_max_abs,
            "note": (
                "Reported test-RMSE-std matches numpy.std(per_fold_curve, ddof=1) "
                "and differs from ddof=0 => sample std."
            ),
        },
        "columns": {
            "test-RMSE-mean": test_mean.tolist(),
            "test-RMSE-std": test_std.tolist(),
            "train-RMSE-mean": train_mean.tolist(),
            "train-RMSE-std": train_std.tolist(),
        },
    }
    (CV_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"wrote cv fixtures under {CV_DIR}")
    print(f"  parity self-check: 0 mismatches across {N_SPLITS} folds")
    print(
        f"  ddof: sample(ddof=1) max|diff|={std_ddof1_max_abs:.3e}  "
        f"ddof0 max|diff|={std_ddof0_max_abs:.3e} (>1e-6, differs)"
    )
    print(f"  test-RMSE-mean = {test_mean}")


if __name__ == "__main__":
    main()
