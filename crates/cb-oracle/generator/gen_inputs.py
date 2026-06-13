#!/usr/bin/env python3
"""Generate the FROZEN shared INPUT corpus for the catboost-rs oracle harness.

This is a BUILD-TIME tool (D-12): it runs OUTSIDE CI on a developer machine and
writes committed, frozen fixtures. CI only READS the committed `.npy`/`config.json`
artifacts; it never installs catboost/numpy nor runs this script.

Three datasets are synthesized deterministically (fixed seed) in the hybrid
fixture format (D-09): per-dataset `config.json` metadata alongside `np.float64`
`.npy` arrays.

    fixtures/inputs/numeric_tiny/        ~50 x 4 pure numeric
    fixtures/inputs/numeric_categorical/ ~50 rows x (3 numeric + 2 categorical)
    fixtures/inputs/grouped_ranking/     ~60 rows with group_id

Run:
    cd crates/cb-oracle/generator
    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python gen_inputs.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Repo-relative fixtures root: <repo>/crates/cb-oracle/fixtures
FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
INPUTS = FIXTURES / "inputs"

# Fixed master seed for input synthesis (recorded in each config.json).
INPUT_SEED = 20260613


def _assert_f64(arr: np.ndarray, name: str) -> np.ndarray:
    """Pitfall 3: a silent f32 downcast would shift the 1e-5 baseline."""
    if arr.dtype != np.float64:
        raise AssertionError(f"{name} must be np.float64, got {arr.dtype}")
    return arr


def _write(dataset_dir: Path, arrays: dict[str, np.ndarray], config: dict) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for stem, arr in arrays.items():
        np.save(dataset_dir / f"{stem}.npy", arr, allow_pickle=False)
    with (dataset_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
        fh.write("\n")


def gen_numeric_tiny() -> None:
    rng = np.random.default_rng(INPUT_SEED)
    n_rows, n_cols = 50, 4
    x = _assert_f64(rng.standard_normal((n_rows, n_cols)).astype(np.float64), "numeric_tiny X")
    # Linear-ish regression target with mild noise; pure f64.
    weights = np.array([1.5, -2.0, 0.5, 3.0], dtype=np.float64)
    y = _assert_f64((x @ weights + 0.1 * rng.standard_normal(n_rows)).astype(np.float64), "numeric_tiny y")
    _write(
        INPUTS / "numeric_tiny",
        {"X": x, "y": y},
        {
            "dataset": "numeric_tiny",
            "seed": INPUT_SEED,
            "n_rows": n_rows,
            "n_numeric": n_cols,
            "n_categorical": 0,
            "has_group_id": False,
            "column_kinds": {f"f{i}": "numeric" for i in range(n_cols)},
            "target": "regression",
            "dtype": "float64",
        },
    )


def gen_numeric_categorical() -> None:
    rng = np.random.default_rng(INPUT_SEED + 1)
    n_rows = 50
    n_numeric, n_categorical = 3, 2
    x_num = _assert_f64(rng.standard_normal((n_rows, n_numeric)).astype(np.float64), "numeric_categorical X")
    # Categorical columns encoded as small integer category ids stored as f64
    # (the Rust reader treats them as category labels per config.json).
    cat = rng.integers(low=0, high=4, size=(n_rows, n_categorical)).astype(np.float64)
    cat = _assert_f64(cat, "numeric_categorical cat")
    weights = np.array([2.0, -1.0, 0.75], dtype=np.float64)
    cat_effect = cat[:, 0] * 0.5 - cat[:, 1] * 0.3
    y = _assert_f64(
        (x_num @ weights + cat_effect + 0.1 * rng.standard_normal(n_rows)).astype(np.float64),
        "numeric_categorical y",
    )
    _write(
        INPUTS / "numeric_categorical",
        {"X": x_num, "cat": cat, "y": y},
        {
            "dataset": "numeric_categorical",
            "seed": INPUT_SEED + 1,
            "n_rows": n_rows,
            "n_numeric": n_numeric,
            "n_categorical": n_categorical,
            "cat_feature_indices": [n_numeric, n_numeric + 1],
            "has_group_id": False,
            "column_kinds": {
                **{f"f{i}": "numeric" for i in range(n_numeric)},
                **{f"c{i}": "categorical" for i in range(n_categorical)},
            },
            "target": "regression",
            "dtype": "float64",
        },
    )


def gen_grouped_ranking() -> None:
    rng = np.random.default_rng(INPUT_SEED + 2)
    n_rows = 60
    n_cols = 3
    x = _assert_f64(rng.standard_normal((n_rows, n_cols)).astype(np.float64), "grouped_ranking X")
    # 12 groups of 5 rows each (60 total); group_id ascending so groups are
    # contiguous (CatBoost ranking requirement).
    group_size = 5
    n_groups = n_rows // group_size
    group_id = np.repeat(np.arange(n_groups, dtype=np.int64), group_size).astype(np.float64)
    group_id = _assert_f64(group_id, "grouped_ranking group_id")
    weights = np.array([1.0, 0.5, -1.5], dtype=np.float64)
    y = _assert_f64((x @ weights + 0.1 * rng.standard_normal(n_rows)).astype(np.float64), "grouped_ranking y")
    _write(
        INPUTS / "grouped_ranking",
        {"X": x, "group_id": group_id, "y": y},
        {
            "dataset": "grouped_ranking",
            "seed": INPUT_SEED + 2,
            "n_rows": n_rows,
            "n_numeric": n_cols,
            "n_categorical": 0,
            "has_group_id": True,
            "n_groups": n_groups,
            "group_size": group_size,
            "column_kinds": {f"f{i}": "numeric" for i in range(n_cols)},
            "target": "ranking",
            "dtype": "float64",
        },
    )


def main() -> None:
    gen_numeric_tiny()
    gen_numeric_categorical()
    gen_grouped_ranking()
    print(f"Wrote frozen input corpus under {INPUTS}")


if __name__ == "__main__":
    main()
