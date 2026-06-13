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
    fixtures/inputs/numeric_nan/         ~50 x 3 numeric with NaN entries (Wave-0 A1/A3)
    fixtures/inputs/explicit_categorical/ ~30 rows of EXPLICIT string cat columns (Wave-0 A4)

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


def gen_numeric_nan() -> None:
    """NaN-containing numeric dataset (Wave-0). Exercises the NanMode sentinel
    path (DATA-04) so the borders oracle can resolve A1/A3 (whether
    `get_borders()` surfaces the `f32::MIN`/`MAX` sentinel for a NaN feature).
    Column 0 carries `np.nan` in known positions; columns 1-2 are NaN-free."""
    rng = np.random.default_rng(INPUT_SEED + 3)
    n_rows, n_cols = 50, 3
    x = _assert_f64(rng.standard_normal((n_rows, n_cols)).astype(np.float64), "numeric_nan X")
    # Inject NaN into column 0 at deterministic, recorded row positions.
    nan_rows = [3, 7, 11, 19, 28, 41]
    for r in nan_rows:
        x[r, 0] = np.nan
    # Target must not be NaN (CatBoost rejects NaN labels); derive from the
    # NaN-free columns only so y stays finite.
    weights = np.array([1.0, -0.5], dtype=np.float64)
    y = _assert_f64(
        (x[:, 1:] @ weights + 0.1 * rng.standard_normal(n_rows)).astype(np.float64),
        "numeric_nan y",
    )
    _write(
        INPUTS / "numeric_nan",
        {"X": x, "y": y},
        {
            "dataset": "numeric_nan",
            "seed": INPUT_SEED + 3,
            "n_rows": n_rows,
            "n_numeric": n_cols,
            "n_categorical": 0,
            "has_group_id": False,
            "nan_feature_index": 0,
            "nan_row_indices": nan_rows,
            "column_kinds": {f"f{i}": "numeric" for i in range(n_cols)},
            "target": "regression",
            "dtype": "float64",
            "note": (
                "Column 0 contains np.nan at nan_row_indices; columns 1-2 are "
                "NaN-free. Used by the borders_quant oracle to resolve A1/A3 "
                "(NanMode sentinel presence in get_borders())."
            ),
        },
    )


def gen_explicit_categorical() -> None:
    """Explicit STRING categorical dataset (Wave-0). Categories are fed as plain
    strings (NOT f64-coded), removing the integer-stringification ambiguity
    (A4). The `categories` list records first-seen iteration order so the
    perfect-hash remap (first-seen -> bin) can be validated end-to-end. A `num`
    column keeps the feature set non-constant for training."""
    rng = np.random.default_rng(INPUT_SEED + 4)
    n_rows = 30
    # Deterministic explicit string categories with a known first-seen order.
    # First-seen order over c0: alpha, beta, gamma, delta, epsilon.
    c0_pool = ["alpha", "beta", "gamma", "delta", "epsilon"]
    c0 = [c0_pool[i % len(c0_pool)] for i in range(n_rows)]
    # c1 mixes integer-valued strings to stress A4 ("3" vs "3.0"): plain ints.
    c1_pool = ["3", "10", "-2", "7"]
    c1 = [c1_pool[(i * 3) % len(c1_pool)] for i in range(n_rows)]
    num = _assert_f64(rng.standard_normal(n_rows).astype(np.float64), "explicit_categorical num")
    y = (rng.integers(low=0, high=2, size=n_rows)).astype(np.int64)

    def first_seen(seq: list[str]) -> list[str]:
        return list(dict.fromkeys(seq))

    dataset_dir = INPUTS / "explicit_categorical"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    # Categorical columns are strings -> store as .npy object-free via a JSON
    # sidecar (np.save of <U strings is allowed; load with allow_pickle=False).
    np.save(dataset_dir / "c0.npy", np.asarray(c0, dtype="U16"), allow_pickle=False)
    np.save(dataset_dir / "c1.npy", np.asarray(c1, dtype="U16"), allow_pickle=False)
    np.save(dataset_dir / "num.npy", num, allow_pickle=False)
    np.save(dataset_dir / "y.npy", y, allow_pickle=False)
    config = {
        "dataset": "explicit_categorical",
        "seed": INPUT_SEED + 4,
        "n_rows": n_rows,
        "n_numeric": 1,
        "n_categorical": 2,
        "cat_feature_columns": ["c0", "c1"],
        "numeric_columns": ["num"],
        "cat_features_are_strings": True,
        "c0_first_seen_order": first_seen(c0),
        "c1_first_seen_order": first_seen(c1),
        "has_group_id": False,
        "column_kinds": {"num": "numeric", "c0": "categorical", "c1": "categorical"},
        "target": "binary",
        "dtype_numeric": "float64",
        "dtype_categorical": "string",
        "note": (
            "Categorical columns are EXPLICIT strings (resolves A4). c1 uses "
            "integer-valued strings ('3','10','-2','7') stringified as plain "
            "integers (no '.0'), matching CatBoost CalcCatFeatureHash on the "
            "string form. First-seen orders are recorded for perfect-hash "
            "(first-seen -> bin) validation."
        ),
    }
    with (dataset_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
        fh.write("\n")


def main() -> None:
    gen_numeric_tiny()
    gen_numeric_categorical()
    gen_grouped_ranking()
    gen_numeric_nan()
    gen_explicit_categorical()
    print(f"Wrote frozen input corpus under {INPUTS}")


if __name__ == "__main__":
    main()
