#!/usr/bin/env python3
"""Generate per-stage expected-OUTPUT oracle fixtures (regression_skeleton).

BUILD-TIME tool (D-12): runs OUTSIDE CI, writes committed frozen fixtures. CI
only READS the committed artifacts.

Loads the frozen `numeric_tiny` INPUT corpus (produced by `gen_inputs.py`),
trains a pinned `CatBoostRegressor`, and extracts the per-stage oracle values the
Rust harness compares against at <= 1e-5 (INFRA-04):

    get_borders()            -> borders.npy        (flattened f64, layout below)
    save_model(format=json)  -> model.json         (splits + leaf_values, INFRA-04)
    staged_predict()         -> staged.npy         (n_iterations x n_rows, flat f64)
    predict()                -> predictions.npy    (n_rows f64)

Determinism (Pitfall 2 / T-01-07): thread_count=1 and a fixed random_seed make
the summation order reproducible; the exact params + seed are recorded in
fixtures/regression_skeleton/config.json so the baseline is auditable.

borders.npy layout: borders for feature 0, then feature 1, ... concatenated in
ascending feature-index order. borders_per_feature.npy records the count for each
feature (f64-encoded integer counts) so a reader can split the flat vector back
into per-feature border lists.

Run (after gen_inputs.py):
    .venv/bin/python gen_fixtures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
INPUTS = FIXTURES / "inputs"
SCENARIO = FIXTURES / "regression_skeleton"
BORDERS_QUANT = FIXTURES / "borders_quant"
CAT_HASH = FIXTURES / "cat_hash"
CLASS_WEIGHTS = FIXTURES / "class_weights"

# IEEE-754 single-precision extremes — the NanMode sentinel borders upstream
# injects into the STORED border set (quantization.cpp:342/344). Wave-0 probing
# (A3) shows get_borders() STRIPS these, so they never appear in the oracle.
F32_MIN = float(np.finfo(np.float32).min)
F32_MAX = float(np.finfo(np.float32).max)

CATBOOST_VERSION = "1.2.10"
SEED = 0
PARAMS = {
    "iterations": 10,
    "learning_rate": 0.1,
    "depth": 4,
    "random_seed": SEED,
    "thread_count": 1,  # Pitfall 2: pin to 1 for deterministic summation order.
    "verbose": False,
}


def _assert_f64(arr: np.ndarray, name: str) -> np.ndarray:
    if arr.dtype != np.float64:
        raise AssertionError(f"{name} must be np.float64, got {arr.dtype}")
    return arr


def _extract_borders(model) -> tuple[np.ndarray, np.ndarray, list[int], bool]:
    """Flatten get_borders() into (flat_borders, per_feature_counts, indices,
    sentinel_present). Layout matches regression_skeleton: feature 0 borders,
    then feature 1, ... ascending. sentinel_present records whether any returned
    border equals the f32 MIN/MAX NanMode sentinel (A3)."""
    borders_dict = model.get_borders()
    indices = sorted(borders_dict.keys())
    flat: list[float] = []
    counts: list[float] = []
    sentinel = False
    for fi in indices:
        fb = list(borders_dict[fi])
        for b in fb:
            bf = float(b)
            if bf <= F32_MIN * 0.99 or bf >= F32_MAX * 0.99:
                sentinel = True
        flat.extend(float(b) for b in fb)
        counts.append(float(len(fb)))
    return (
        _assert_f64(np.asarray(flat, dtype=np.float64), "borders"),
        _assert_f64(np.asarray(counts, dtype=np.float64), "borders_per_feature"),
        indices,
        sentinel,
    )


def gen_borders_quant() -> None:
    """borders_quant/ — border/quantization oracle for numeric_tiny (NaN-free)
    and numeric_nan (NanMode sentinel path). Resolves A1/A2/A3: records the
    default border_count, nan_mode, GreedyLogSum selection, and EMPIRICALLY
    whether get_borders() surfaces the NaN sentinel."""
    BORDERS_QUANT.mkdir(parents=True, exist_ok=True)
    scenarios = {}
    for name in ("numeric_tiny", "numeric_nan"):
        x = np.load(INPUTS / name / "X.npy")
        y = np.load(INPUTS / name / "y.npy")
        # Defaults EXCEPT thread_count=1 / pinned seed so we read upstream's
        # default border_count (A2) and nan_mode back from the trained model.
        model = CatBoostRegressor(
            iterations=10, random_seed=SEED, thread_count=1, verbose=False
        )
        model.fit(x, y)
        flat, counts, indices, sentinel = _extract_borders(model)
        np.save(BORDERS_QUANT / f"{name}.borders.npy", flat, allow_pickle=False)
        np.save(
            BORDERS_QUANT / f"{name}.borders_per_feature.npy", counts, allow_pickle=False
        )
        ap = model.get_all_params()
        scenarios[name] = {
            "input_dataset": name,
            "border_count": ap.get("border_count"),
            "nan_mode": ap.get("nan_mode"),
            "border_selection_type": ap.get("feature_border_type"),
            "borders_feature_indices": indices,
            "nan_sentinel_present_in_get_borders": sentinel,
            "n_borders_total": int(flat.shape[0]),
        }
    config = {
        "scenario": "borders_quant",
        "seed": SEED,
        "catboost_version": CATBOOST_VERSION,
        "thread_count": 1,
        "scenarios": scenarios,
        "borders_layout": (
            "<dataset>.borders.npy = flat f64 (feature 0 borders, then feature "
            "1, ...); <dataset>.borders_per_feature.npy = per-feature counts."
        ),
        "A1_A3_resolution": (
            "EMPIRICAL (catboost 1.2.10): get_borders() DOES surface the NanMode "
            "f32::MIN sentinel for a NaN feature under nan_mode=Min, as "
            "borders[0] = -3.4028234663852886e+38 "
            "(== numeric_limits<float>::lowest). This scenario uses default "
            "params (border_count=254, nan_mode=Min) and records "
            "scenarios.numeric_nan.nan_sentinel_present_in_get_borders=true; the "
            "NaN-free numeric_tiny set records =false. CAVEAT (config-dependent, "
            "verified): sentinel inclusion tracks the realized border budget, "
            "not nan_mode alone -- at a small border budget (e.g. depth=4 / few "
            "borders) the same NaN(Min) feature can OMIT the sentinel (4-5 "
            "borders, borders[0] != f32::MIN), and nan_mode=Max never prepends "
            "f32::MIN. The Rust border oracle MUST therefore compare against the "
            "per-fixture borders verbatim (sentinel present iff the fixture has "
            "it at index 0) rather than assuming a fixed rule. This fixture pins "
            "the default-param baseline (sentinel PRESENT)."
        ),
        "A2_resolution": (
            "Default border_count in catboost 1.2.10 is recorded per scenario "
            "(observed: 254), border_selection_type=GreedyLogSum, nan_mode=Min."
        ),
    }
    with (BORDERS_QUANT / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _isolate_cat_hash(probe: str, anchors: list[str]) -> int:
    """Return the full ui64 CalcCatFeatureHash precursor for `probe` by training
    a CTR model on (anchors + [probe]) and taking the hash_map entry absent from
    an anchors-only run. Deterministic; thread_count=1."""
    empty = 18446744073709551615

    def hash_set(distinct: list[str]) -> set[int]:
        import pandas as pd

        cats = (distinct * 6)[: max(12, len(distinct) * 3)]
        labels = [j % 2 for j in range(len(cats))]
        df = pd.DataFrame({"c0": cats})
        pool = Pool(df, label=labels, cat_features=["c0"])
        m = CatBoostClassifier(
            iterations=20, depth=2, random_seed=SEED, thread_count=1, verbose=False
        )
        m.fit(pool)
        import tempfile

        p = tempfile.mktemp(suffix=".json")
        m.save_model(p, format="json")
        cd = json.load(open(p)).get("ctr_data") or {}
        key = next(k for k in cd if '"type":"Borders"' in k)
        hm = cd[key]["hash_map"]
        st = cd[key]["hash_stride"]
        return {int(hm[i]) for i in range(0, len(hm), st) if int(hm[i]) != empty}

    base = hash_set(anchors)
    full = hash_set(anchors + [probe])
    new = [h for h in full if h not in base]
    if len(new) != 1:
        raise AssertionError(f"hash isolation for {probe!r} ambiguous: {sorted(new)}")
    return new[0]


def gen_cat_hash() -> None:
    """cat_hash/ — bit-exact (cat string -> ui32) CalcCatFeatureHash vectors and
    perfect-hash (first-seen -> bin) assignment for the explicit_categorical
    corpus. Resolves A4 (integer cats stringify as plain integers: '3' != '3.0')
    and A5 (hash source = upstream-extracted, not a third-party crate)."""
    CAT_HASH.mkdir(parents=True, exist_ok=True)
    cfg = json.load((INPUTS / "explicit_categorical" / "config.json").open())
    c0 = [str(v) for v in np.load(INPUTS / "explicit_categorical" / "c0.npy")]
    c1 = [str(v) for v in np.load(INPUTS / "explicit_categorical" / "c1.npy")]

    # Stable anchor set kept distinct from all probe strings so the CTR builds.
    anchors = ["A_anchor0", "A_anchor1", "A_anchor2", "A_anchor3"]
    # All distinct category strings across both columns, first-seen order.
    distinct = list(dict.fromkeys(c0 + c1))
    # A4 demonstrator: include '3' and '3.0' to PROVE they hash differently.
    a4_demo = ["3", "3.0"]
    probe_strings = list(dict.fromkeys(distinct + a4_demo))

    str_to_ui64: dict[str, int] = {}
    for s in probe_strings:
        str_to_ui64[s] = _isolate_cat_hash(s, anchors)
    str_to_ui32 = {s: (h & 0xFFFFFFFF) for s, h in str_to_ui64.items()}

    # Per-object ui32 hash + perfect-hash bin (first-seen -> incrementing bin)
    # for each categorical column, mirroring upstream first-seen remap
    # (cat_feature_perfect_hash_helper.cpp:120).
    def per_object(col: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
        order: dict[str, int] = {}
        bins = []
        hashes = []
        for v in col:
            if v not in order:
                order[v] = len(order)
            bins.append(float(order[v]))
            hashes.append(float(str_to_ui32[v]))
        return (
            _assert_f64(np.asarray(hashes, dtype=np.float64), "cat_hashes"),
            _assert_f64(np.asarray(bins, dtype=np.float64), "perfect_hash_bins"),
            list(order.keys()),
        )

    c0_h, c0_bins, c0_order = per_object(c0)
    c1_h, c1_bins, c1_order = per_object(c1)
    # Combined per-object arrays: column 0 then column 1, concatenated.
    np.save(
        CAT_HASH / "cat_hashes.npy",
        _assert_f64(np.concatenate([c0_h, c1_h]), "cat_hashes"),
        allow_pickle=False,
    )
    np.save(
        CAT_HASH / "perfect_hash_bins.npy",
        _assert_f64(np.concatenate([c0_bins, c1_bins]), "perfect_hash_bins"),
        allow_pickle=False,
    )

    config = {
        "scenario": "cat_hash",
        "seed": SEED,
        "catboost_version": CATBOOST_VERSION,
        "thread_count": 1,
        "input_dataset": "explicit_categorical",
        "hash_definition": "CalcCatFeatureHash(s) = CityHash64(s) & 0xffffffff",
        "string_to_ui32": {s: int(u) for s, u in sorted(str_to_ui32.items())},
        "string_to_ui64_precursor": {
            s: str(h) for s, h in sorted(str_to_ui64.items())
        },
        "c0_first_seen_order": c0_order,
        "c1_first_seen_order": c1_order,
        "cat_hashes_layout": (
            "flat f64-encoded ui32: column c0 per-object hashes (n_rows), then "
            "column c1 per-object hashes (n_rows)."
        ),
        "perfect_hash_bins_layout": (
            "flat f64-encoded bin index: column c0 bins (n_rows), then c1 bins "
            "(n_rows); bin = first-seen order within each column."
        ),
        "A4_resolution": (
            "Integer cat features stringify as PLAIN integers before hashing: "
            f"'3' -> ui32={str_to_ui32.get('3')} differs from "
            f"'3.0' -> ui32={str_to_ui32.get('3.0')}. Rust must hash the "
            "integer string form ('3'), never the float form ('3.0')."
        ),
        "A5_resolution": (
            "(string -> ui32) vectors are EXTRACTED from upstream catboost "
            "1.2.10 (model.json ctr_data hash_map), not from a third-party "
            "cityhash crate. Rust's port of util/digest/city.cpp must reproduce "
            "string_to_ui32 bit-exactly."
        ),
    }
    with (CAT_HASH / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
        fh.write("\n")


def gen_class_weights() -> None:
    """class_weights/ — Balanced and SqrtBalanced auto class-weight oracle on an
    imbalanced binary-label dataset. Balanced = max/count, SqrtBalanced =
    sqrt(max/count) (DATA-08), floor 1e-8, summary sums accumulated in double.
    Values are read back from CatBoost's own computed `class_weights` param so
    the oracle is upstream-authoritative."""
    CLASS_WEIGHTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    # Imbalanced binary labels: 30 of class 0, 10 of class 1.
    y = np.array([0] * 30 + [1] * 10, dtype=np.int64)
    n = y.shape[0]
    x = _assert_f64(rng.standard_normal((n, 3)).astype(np.float64), "class_weights X")
    counts = np.bincount(y).astype(np.float64)

    computed = {}
    for acw, stem in (("Balanced", "balanced"), ("SqrtBalanced", "sqrt_balanced")):
        m = CatBoostClassifier(
            iterations=2,
            depth=2,
            random_seed=SEED,
            thread_count=1,
            verbose=False,
            auto_class_weights=acw,
        )
        m.fit(x, y)
        weights = m.get_all_params().get("class_weights")
        arr = _assert_f64(np.asarray(weights, dtype=np.float64), f"{acw} weights")
        np.save(CLASS_WEIGHTS / f"{stem}.npy", arr, allow_pickle=False)
        computed[acw] = [float(w) for w in weights]

    config = {
        "scenario": "class_weights",
        "seed": SEED,
        "catboost_version": CATBOOST_VERSION,
        "thread_count": 1,
        "class_counts": [float(c) for c in counts],
        "n_classes": int(counts.shape[0]),
        "formulas": {
            "Balanced": "max(counts) / counts[c]",
            "SqrtBalanced": "sqrt(max(counts) / counts[c])",
            "floor": 1e-8,
            "accumulation": "double (f64) summary sums",
        },
        "computed": computed,
        "source": "CatBoostClassifier.get_all_params()['class_weights'] (upstream-computed)",
    }
    with (CLASS_WEIGHTS / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
        fh.write("\n")


def main() -> None:
    # Load the frozen numeric_tiny input corpus.
    x = np.load(INPUTS / "numeric_tiny" / "X.npy")
    y = np.load(INPUTS / "numeric_tiny" / "y.npy")

    model = CatBoostRegressor(**PARAMS)
    model.fit(x, y)

    SCENARIO.mkdir(parents=True, exist_ok=True)

    # --- Stage: Borders -----------------------------------------------------
    # get_borders() -> dict {feature_index: [border, ...]} in 1.2.10.
    borders_dict = model.get_borders()
    feature_indices = sorted(borders_dict.keys())
    flat_borders: list[float] = []
    counts: list[float] = []
    for fi in feature_indices:
        feat_borders = list(borders_dict[fi])
        flat_borders.extend(float(b) for b in feat_borders)
        counts.append(float(len(feat_borders)))
    borders_arr = _assert_f64(np.asarray(flat_borders, dtype=np.float64), "borders")
    counts_arr = _assert_f64(np.asarray(counts, dtype=np.float64), "borders_per_feature")
    np.save(SCENARIO / "borders.npy", borders_arr, allow_pickle=False)
    np.save(SCENARIO / "borders_per_feature.npy", counts_arr, allow_pickle=False)

    # --- Stage: Splits + LeafValues (model.json) ----------------------------
    model.save_model(str(SCENARIO / "model.json"), format="json")

    # --- Stage: StagedApprox ------------------------------------------------
    staged = [np.asarray(p, dtype=np.float64) for p in model.staged_predict(x)]
    staged_flat = _assert_f64(
        np.concatenate([s.ravel() for s in staged]).astype(np.float64), "staged"
    )
    np.save(SCENARIO / "staged.npy", staged_flat, allow_pickle=False)

    # --- Stage: Predictions -------------------------------------------------
    predictions = _assert_f64(np.asarray(model.predict(x), dtype=np.float64), "predictions")
    np.save(SCENARIO / "predictions.npy", predictions, allow_pickle=False)

    # --- config.json (hybrid fixture metadata, D-09) ------------------------
    config = {
        "scenario": "regression_skeleton",
        "seed": SEED,
        "catboost_version": CATBOOST_VERSION,
        "thread_count": 1,
        "input_dataset": "numeric_tiny",
        "params": PARAMS,
        "n_rows": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "n_iterations": len(staged),
        "borders_feature_indices": feature_indices,
        "stages": ["Borders", "Splits", "LeafValues", "StagedApprox", "Predictions"],
        "borders_layout": (
            "flat f64: feature 0 borders, then feature 1, ... ; "
            "borders_per_feature.npy gives the per-feature counts"
        ),
        "staged_layout": "flat f64: stage 0 (n_rows), then stage 1, ... ; n_iterations stages",
    }
    with (SCENARIO / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(f"Wrote regression_skeleton oracle fixtures under {SCENARIO}")
    print(
        f"  borders={borders_arr.shape}, staged={staged_flat.shape}, "
        f"predictions={predictions.shape}"
    )

    # --- Wave-0 scenarios (A1-A5 resolution) --------------------------------
    gen_borders_quant()
    print(f"Wrote borders_quant oracle fixtures under {BORDERS_QUANT}")
    gen_cat_hash()
    print(f"Wrote cat_hash oracle fixtures under {CAT_HASH}")
    gen_class_weights()
    print(f"Wrote class_weights oracle fixtures under {CLASS_WEIGHTS}")


if __name__ == "__main__":
    main()
