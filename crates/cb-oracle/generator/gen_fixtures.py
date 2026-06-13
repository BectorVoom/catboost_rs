#!/usr/bin/env python3
"""Generate per-stage expected-OUTPUT oracle fixtures (training + Wave-0).

RUN-ONCE / COMMIT discipline (D-12): this is a BUILD-TIME tool that runs OUTSIDE
CI and writes committed frozen fixtures. CI only READS the committed artifacts;
the generator is NEVER invoked from CI. Re-run it by hand only to regenerate a
fixture, then COMMIT the result. It adds NO C++ build step for the training
oracles (D-11 — Python-reachable oracle only; the cat_hash scenario's small C++
cityhash oracle is a pre-existing Wave-0 asset unrelated to training).

Loads the frozen `numeric_tiny` INPUT corpus (produced by `gen_inputs.py`) and
trains TWO pinned training oracles whose per-stage values the Rust harness
compares against at <= 1e-5 (INFRA-04):

    regression_skeleton (RMSE)   -> CatBoostRegressor,  boost_from_average=True
    binclf_skeleton     (Logloss) -> CatBoostClassifier, boost_from_average=False

For each scenario:
    save_model(format=json)  -> model.json    (splits + leaf_values, INFRA-04)
    staged_predict()         -> staged.npy    (n_iterations x n_rows, flat f64;
                                                Logloss uses RawFormulaVal raw
                                                logits, A5/Pitfall 6)
    predict()                -> predictions.npy

SIMPLIFIED ISOLATING PARAMS (D-07/A1/A2/A4): both scenarios deliberately pin
`bootstrap_type='No'`, `random_strength=0`, fixed `l2_leaf_reg`/`depth`/
`learning_rate`/`iterations`, `leaf_estimation_iterations=1`, `score_function='L2'`
(Open Q1 RESOLVED to L2 — simplest first-slice math), `leaf_estimation_method=
'Gradient'`, a fixed `random_seed`, and `thread_count=1` so any first-slice
divergence can only be the tree/leaf math, not an interacting subsystem.
`boost_from_average` is set explicitly per loss (True for RMSE, False for
Logloss — Pitfall 2).

Determinism (Pitfall 2 / T-01-07): thread_count=1 and a fixed random_seed make
the summation order reproducible; the exact params + seed are recorded in each
scenario's config.json so the baseline is auditable.

Run (after gen_inputs.py):
    .venv/bin/python gen_fixtures.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

GENERATOR_DIR = Path(__file__).resolve().parent
FIXTURES = GENERATOR_DIR.parent / "fixtures"
INPUTS = FIXTURES / "inputs"
SCENARIO = FIXTURES / "regression_skeleton"
BINCLF_SCENARIO = FIXTURES / "binclf_skeleton"
BORDERS_QUANT = FIXTURES / "borders_quant"
CAT_HASH = FIXTURES / "cat_hash"
CLASS_WEIGHTS = FIXTURES / "class_weights"
LEAF_METHODS = FIXTURES / "leaf_methods"
BOOTSTRAP = FIXTURES / "bootstrap"
BOOTSTRAP_INPUT = INPUTS / "bootstrap_multiblock"

# IEEE-754 single-precision extremes — the NanMode sentinel borders upstream
# injects into the STORED border set (quantization.cpp:342/344). Wave-0 probing
# (A3) shows get_borders() STRIPS these, so they never appear in the oracle.
F32_MIN = float(np.finfo(np.float32).min)
F32_MAX = float(np.finfo(np.float32).max)

CATBOOST_VERSION = "1.2.10"
SEED = 0
# Dedicated synthesis seed for the TRAIN-04 multi-block bootstrap dataset (kept
# distinct from numeric_tiny's seed so the corpus is reproducible in isolation).
INPUT_SEED_BOOTSTRAP = 20260613

# D-07 simplified isolating params shared by BOTH training scenarios. Every knob
# that interacts with the tree/leaf math is pinned to its most isolating value so
# a first-slice divergence can only be the tree/leaf math itself (A1/A2/A4):
#   bootstrap_type=No / subsample-free      -> no sampling draw interaction
#   random_strength=0                       -> no score noise (TRAIN-05 deferred)
#   l2_leaf_reg=3.0                         -> fixed, explicit regularization
#   depth=2                                 -> 4 leaves/tree, small split count
#   learning_rate=0.1, iterations=5         -> small, fixed boosting loop
#   leaf_estimation_iterations=1            -> single Newton/Gradient leaf step (A4/Pitfall 5)
#   score_function=L2                       -> simplest split-score math (Open Q1 RESOLVED)
#   leaf_estimation_method=Gradient         -> simplest leaf estimator first
#   random_seed fixed, thread_count=1       -> deterministic summation order (Pitfall 2)
# `boost_from_average` is set PER LOSS at the call site (True RMSE / False
# Logloss, Pitfall 2), so it is intentionally absent from this shared dict.
ISOLATING_PARAMS = {
    "iterations": 5,
    "learning_rate": 0.1,
    "depth": 2,
    "l2_leaf_reg": 3.0,
    "bootstrap_type": "No",
    "random_strength": 0,
    "leaf_estimation_iterations": 1,
    "score_function": "L2",
    "leaf_estimation_method": "Gradient",
    "random_seed": SEED,
    "thread_count": 1,  # Pitfall 2: pin to 1 for deterministic summation order.
    "verbose": False,
}

# Back-compat alias: the borders_quant / numeric_tiny Wave-0 borders stage still
# reads `PARAMS` for the regression skeleton's recorded baseline.
PARAMS = ISOLATING_PARAMS


def _assert_f64(arr: np.ndarray, name: str) -> np.ndarray:
    if arr.dtype != np.float64:
        raise AssertionError(f"{name} must be np.float64, got {arr.dtype}")
    return arr


def _quantization_borders(x, y, nan_mode: str) -> dict[int, list[float]]:
    """Raw GreedyLogSum quantization borders via the STANDALONE binarizer
    (`Pool.quantize(...).save_quantization_borders(...)`), NOT a trained model's
    `get_borders()`.

    This is the parity target the Rust `borders.rs` reproduces: the full
    GreedyLogSum border set at the given `border_count`, including the NanMode
    sentinel for NaN features under `nan_mode=Min`. A trained model's
    `get_borders()` instead returns a *pruned* subset (only borders used by some
    tree split), which is training-dependent and NOT reproducible by a
    standalone binarizer — using it as the oracle target was the Wave-0 bug this
    function fixes (Rule 1)."""
    import tempfile

    pool = Pool(x, y)
    pool.quantize(
        border_count=254,
        feature_border_type="GreedyLogSum",
        nan_mode=nan_mode,
    )
    path = tempfile.mktemp(suffix=".borders.tsv")
    pool.save_quantization_borders(path)
    borders: dict[int, list[float]] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            # Lines are "<feature>\t<border>" for plain features and
            # "<feature>\t<border>\t<nan_mode>" for NaN features; take the first
            # two columns and ignore any trailing nan-mode annotation.
            parts = line.split("\t")
            fi_str, val_str = parts[0], parts[1]
            borders.setdefault(int(fi_str), []).append(float(val_str))
    return borders


def _extract_borders(borders_dict) -> tuple[np.ndarray, np.ndarray, list[int], bool]:
    """Flatten a {feature_index: [borders]} dict into (flat_borders,
    per_feature_counts, indices, sentinel_present). Layout matches
    regression_skeleton: feature 0 borders, then feature 1, ... ascending.
    sentinel_present records whether any returned border equals the f32 MIN/MAX
    NanMode sentinel (A3)."""
    indices = sorted(borders_dict.keys())
    flat: list[float] = []
    counts: list[float] = []
    sentinel = False
    for fi in indices:
        fb = sorted(float(b) for b in borders_dict[fi])
        # save_quantization_borders() serializes the f32 sentinel as truncated
        # text (~10 sig figs), which loses precision vs the exact
        # numeric_limits<float>::lowest the Rust port emits (f32::MIN widened to
        # f64). Snap any sentinel-magnitude border to the EXACT F32_MIN / F32_MAX
        # so the committed f64 oracle equals f64::from(f32::MIN) bit-for-bit.
        snapped: list[float] = []
        for bf in fb:
            if bf <= F32_MIN * 0.99:
                sentinel = True
                snapped.append(F32_MIN)
            elif bf >= F32_MAX * 0.99:
                sentinel = True
                snapped.append(F32_MAX)
            else:
                snapped.append(bf)
        flat.extend(snapped)
        counts.append(float(len(snapped)))
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
    # nan_mode=Min is the catboost 1.2.10 default (A2). numeric_tiny is NaN-free
    # so the sentinel never appears; numeric_nan has a NaN feature so the
    # f32::MIN sentinel is prepended to that feature's borders under Min.
    nan_mode = "Min"
    border_count = 254
    for name in ("numeric_tiny", "numeric_nan"):
        x = np.load(INPUTS / name / "X.npy")
        y = np.load(INPUTS / name / "y.npy")
        # RAW standalone GreedyLogSum quantization borders (the Rust parity
        # target), NOT a trained model's pruned get_borders() (Rule 1 fix).
        borders_dict = _quantization_borders(x, y, nan_mode)
        flat, counts, indices, sentinel = _extract_borders(borders_dict)
        np.save(BORDERS_QUANT / f"{name}.borders.npy", flat, allow_pickle=False)
        np.save(
            BORDERS_QUANT / f"{name}.borders_per_feature.npy", counts, allow_pickle=False
        )
        scenarios[name] = {
            "input_dataset": name,
            "border_count": border_count,
            "nan_mode": nan_mode,
            "border_selection_type": "GreedyLogSum",
            "borders_feature_indices": indices,
            "nan_sentinel_present_in_get_borders": sentinel,
            "borders_source": "standalone Pool.quantize().save_quantization_borders()",
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
            "EMPIRICAL (catboost 1.2.10): the STANDALONE GreedyLogSum "
            "quantization (Pool.quantize().save_quantization_borders()) surfaces "
            "the NanMode f32::MIN sentinel as borders[0] for a NaN feature under "
            "nan_mode=Min (numeric_nan feature 0: borders[0] = "
            "-3.4028234663852886e+38 == numeric_limits<float>::lowest); the "
            "NaN-free numeric_tiny set has no sentinel. nan_mode=Max never "
            "prepends f32::MIN. NOTE (Rule-1 fix, 02-02): these borders are the "
            "RAW standalone quantization output, NOT a trained model's "
            "get_borders() -- the latter returns a training-PRUNED subset (only "
            "borders used by some tree split, ~11/11/7/15 for numeric_tiny) that "
            "no standalone binarizer can reproduce. The Rust borders.rs "
            "reproduces the raw standalone set (49/49/49/49 for numeric_tiny; "
            "44/49/49 for numeric_nan with the feature-0 sentinel) verbatim "
            "(per-feature, sentinel present iff index 0 equals f32::MIN)."
        ),
        "A2_resolution": (
            "border_count=254 (catboost 1.2.10 default), "
            "border_selection_type=GreedyLogSum, nan_mode=Min."
        ),
    }
    with (BORDERS_QUANT / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
        fh.write("\n")


_CITYHASH_ORACLE_SRC = GENERATOR_DIR / "cityhash_oracle.cpp"
_VENDORED_CITY_CPP = (
    GENERATOR_DIR.parent.parent.parent / "catboost-master" / "util" / "digest" / "city.cpp"
)


def _build_cityhash_oracle() -> Path:
    """Compile the standalone CalcCatFeatureHash oracle (a dependency-free
    transcription of the vendored `util/digest/city.cpp` — the same algorithm the
    live catboost library compiles) and return the executable path.

    AUTHORITATIVE SOURCE OF TRUTH (A5 correction): the previous fixtures pulled
    (string -> ui32) values from a trained model's `ctr_data` `hash_map`, which
    stores CTR-PROJECTION hashes (CalcHashes over projections, MultiHash-combined
    with priors — index_hash_calcer.h), NOT raw `CalcCatFeatureHash(string)`.
    Those are the wrong oracle target for a CityHash64 port. We instead hash each
    string with the vendored CityHash 1.0 algorithm directly."""
    cxx = shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        raise RuntimeError(
            "no C++ compiler (g++/clang++) found to build the CalcCatFeatureHash oracle"
        )
    out = Path(tempfile.mkdtemp()) / "cityhash_oracle"
    subprocess.run(
        [cxx, "-O2", "-std=c++17", str(_CITYHASH_ORACLE_SRC), "-o", str(out)],
        check=True,
    )
    return out


def _calc_cat_feature_hashes(strings: list[str]) -> dict[str, tuple[int, int]]:
    """Map each string to (ui64 CityHash64, ui32 CalcCatFeatureHash) via the
    vendored-source oracle tool. Strings are passed one-per-line; they must not
    contain a newline (none of the categorical corpus does)."""
    exe = _build_cityhash_oracle()
    payload = "".join(s + "\n" for s in strings)
    proc = subprocess.run(
        [str(exe)], input=payload, capture_output=True, text=True, check=True
    )
    lines = proc.stdout.splitlines()
    if len(lines) != len(strings):
        raise AssertionError(
            f"oracle returned {len(lines)} lines for {len(strings)} strings"
        )
    out: dict[str, tuple[int, int]] = {}
    for s, line in zip(strings, lines):
        ui64_s, ui32_s = line.split("\t")
        out[s] = (int(ui64_s), int(ui32_s))
    return out


def gen_cat_hash() -> None:
    """cat_hash/ — bit-exact (cat string -> ui32) CalcCatFeatureHash vectors and
    perfect-hash (first-seen -> bin) assignment for the explicit_categorical
    corpus. Resolves A4 (integer cats stringify as plain integers: '3' != '3.0')
    and A5 (hash source = upstream-extracted, not a third-party crate)."""
    CAT_HASH.mkdir(parents=True, exist_ok=True)
    cfg = json.load((INPUTS / "explicit_categorical" / "config.json").open())
    c0 = [str(v) for v in np.load(INPUTS / "explicit_categorical" / "c0.npy")]
    c1 = [str(v) for v in np.load(INPUTS / "explicit_categorical" / "c1.npy")]

    # All distinct category strings across both columns, first-seen order.
    distinct = list(dict.fromkeys(c0 + c1))
    # A4 demonstrator: include '3' and '3.0' to PROVE they hash differently.
    a4_demo = ["3", "3.0"]
    probe_strings = list(dict.fromkeys(distinct + a4_demo))

    # Hash each distinct string with the vendored CityHash 1.0 oracle
    # (CalcCatFeatureHash, NOT the trained model's CTR hash_map — see
    # _build_cityhash_oracle for why the old extraction was wrong).
    str_to_hashes = _calc_cat_feature_hashes(probe_strings)
    str_to_ui64 = {s: h64 for s, (h64, _h32) in str_to_hashes.items()}
    str_to_ui32 = {s: h32 for s, (_h64, h32) in str_to_hashes.items()}

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
            "(string -> ui32) vectors are CalcCatFeatureHash(s) = "
            "CityHash64(s) & 0xffffffff, computed by a standalone C++ oracle "
            "(generator/cityhash_oracle.cpp) transcribed from the VENDORED "
            "catboost-master/util/digest/city.cpp (CityHash 1.0, the same "
            "algorithm the live catboost library compiles). CORRECTION: the "
            "previous fixtures pulled these from a trained model's ctr_data "
            "hash_map, which stores CTR-PROJECTION hashes (CalcHashes + MultiHash "
            "+ priors, index_hash_calcer.h), NOT raw CalcCatFeatureHash -- the "
            "wrong oracle target for a CityHash64 port. Rust's port of "
            "util/digest/city.cpp reproduces string_to_ui32 bit-exactly."
        ),
        "borders_source": "vendored util/digest/city.cpp via generator/cityhash_oracle.cpp",
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


def gen_binclf_skeleton() -> None:
    """binclf_skeleton/ — Logloss binary-classification training oracle mirroring
    regression_skeleton with the SAME simplified isolating params (D-07) except
    `boost_from_average=False` (Pitfall 2: Logloss boosts from 0, not the label
    mean). Reuses the frozen `numeric_tiny` feature matrix; binary labels are the
    deterministic `y > median(y)` split of the same target (no new input corpus).

    staged.npy stores RAW LOGITS via prediction_type='RawFormulaVal' (A5 /
    Pitfall 6) so the oracle is the pre-sigmoid approximant the Rust boosting
    loop produces directly — NOT the sigmoid probability."""
    x = np.load(INPUTS / "numeric_tiny" / "X.npy")
    y_cont = np.load(INPUTS / "numeric_tiny" / "y.npy")
    # Deterministic binary labels from the frozen regression target.
    y = (y_cont > np.median(y_cont)).astype(np.int64)

    model = CatBoostClassifier(boost_from_average=False, **ISOLATING_PARAMS)
    model.fit(x, y)

    BINCLF_SCENARIO.mkdir(parents=True, exist_ok=True)

    # --- Stage: Splits + LeafValues (model.json) ----------------------------
    model.save_model(str(BINCLF_SCENARIO / "model.json"), format="json")

    # --- Stage: StagedApprox (RAW logits, A5/Pitfall 6) ---------------------
    staged = [
        np.asarray(p, dtype=np.float64)
        for p in model.staged_predict(x, prediction_type="RawFormulaVal")
    ]
    staged_flat = _assert_f64(
        np.concatenate([s.ravel() for s in staged]).astype(np.float64), "staged"
    )
    np.save(BINCLF_SCENARIO / "staged.npy", staged_flat, allow_pickle=False)

    # --- Stage: Predictions (raw logits, to match staged final stage) -------
    predictions = _assert_f64(
        np.asarray(
            model.predict(x, prediction_type="RawFormulaVal"), dtype=np.float64
        ),
        "predictions",
    )
    np.save(BINCLF_SCENARIO / "predictions.npy", predictions, allow_pickle=False)

    config = {
        "scenario": "binclf_skeleton",
        "seed": SEED,
        "catboost_version": CATBOOST_VERSION,
        "thread_count": 1,
        "input_dataset": "numeric_tiny",
        "loss_function": "Logloss",
        "label_definition": "y_binary = (numeric_tiny.y > median(numeric_tiny.y))",
        "boost_from_average": False,
        "params": {**ISOLATING_PARAMS, "boost_from_average": False},
        "n_rows": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "n_iterations": len(staged),
        "prediction_type": "RawFormulaVal",
        "stages": ["Splits", "LeafValues", "StagedApprox", "Predictions"],
        "staged_layout": "flat f64: stage 0 (n_rows), then stage 1, ... ; n_iterations stages (raw logits)",
    }
    with (BINCLF_SCENARIO / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
        fh.write("\n")


def gen_leaf_methods() -> None:
    """leaf_methods/{gradient,newton,exact,simple}/ — the TRAIN-03 (D-09) four
    leaf-estimation-method oracle. Each scenario pins ONE
    `leaf_estimation_method` and every other knob at the first-slice simplified
    isolating values, so a divergence is attributable to a single method's leaf
    math.

    TRANSCRIBED LEAF FORMULAS (read from vendored catboost 1.2.10, recorded here
    for the Rust Task-2 implementation):

      online_predictor.h:112-178 + approx_calcer.cpp:482-525
      - CalcAverage(sumDelta, count, scaledL2) = count>0 ? sumDelta/(count+scaledL2) : 0
      - ScaleL2Reg(l2, sumAllWeights, allDocCount) = l2*(sumAllWeights/allDocCount)
      - GRADIENT leaf delta = CalcAverage(SumDer, SumWeights, scaledL2)
      - NEWTON   leaf delta = CalcDeltaNewtonBody(sumDer, sumDer2, l2, sumAllW, docCount)
                             = sumDer / (-sumDer2 + scaledL2)
      - SIMPLE: `CalcLeafDeltasSimple` (approx_calcer.cpp:506-524) dispatches the
        ELeavesEstimation::Simple value through the SAME Gradient branch (the
        `else`/Y_ASSERT(Gradient) path); EMPIRICALLY (probe, catboost 1.2.10)
        Simple leaf values are bit-identical to Gradient for these params (A6
        RESOLVED: Simple == the Gradient leaf-delta formula).
      - EXACT (approx_calcer.cpp:681-704 -> optimal_const_for_loss.h:180-216):
        per leaf, collect residuals r_i = target_i - approx_i (as f32) and weights
        w_i, then leafDelta = CalcOneDimensionalOptimumConstApprox(loss, r, w):
          * MAE / Quantile(alpha=0.5, delta=1e-6): weighted sample quantile
            (quantile.cpp CalcSampleQuantileLinearSearch for <100 samples:
            stable-sort r ascending, accumulate w, return first value where
            sumWeight >= totalWeight*alpha - DBL_EPSILON), then the delta
            adjustment in CalculateWeightedTargetQuantile (alpha=0.5, delta=1e-6):
            q -= delta if (lessWeight + equalWeight*alpha >= needWeight-DBL_EPSILON)
            else q += delta.
      NOTE (parity-critical): Exact is ONLY available upstream for
      Quantile/GroupQuantile/MultiQuantile/MAE/MAPE/... (catboost_options.cpp:346
      rejects Exact for RMSE/Logloss). So the `exact` scenario uses loss_function
      'MAE' (alpha=0.5). Newton is mathematically == Gradient for RMSE (der2==-1
      so -sumDer2==sumWeight), so the `newton` scenario uses Logloss (der2 =
      -p(1-p)) where Newton is genuinely distinct from Gradient.

    The stored model.json `leaf_values` are already learning_rate-scaled (the
    boosting loop multiplies the raw delta by learning_rate); staged.npy stores
    the per-iteration RAW approximant (RawFormulaVal for the classifier)."""
    LEAF_METHODS.mkdir(parents=True, exist_ok=True)

    x = np.load(INPUTS / "numeric_tiny" / "X.npy")
    y = np.load(INPUTS / "numeric_tiny" / "y.npy")
    y_bin = (y > np.median(y)).astype(np.int64)

    # (method, loss, estimator, target, boost_from_average, prediction_type)
    scenarios = [
        ("gradient", "RMSE", "Gradient", y, True, None),
        ("newton", "Logloss", "Newton", y_bin, False, "RawFormulaVal"),
        ("exact", "MAE", "Exact", y, False, None),
        ("simple", "RMSE", "Simple", y, True, None),
    ]

    for name, loss, estimator, target, bfa, pred_type in scenarios:
        scenario_dir = LEAF_METHODS / name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        # All four pin the first-slice simplified isolating params, overriding
        # only leaf_estimation_method (and loss_function / boost_from_average
        # per method's parity requirement above).
        params = {**ISOLATING_PARAMS}
        params["leaf_estimation_method"] = estimator
        params["loss_function"] = loss
        params["boost_from_average"] = bfa

        if loss == "Logloss":
            model = CatBoostClassifier(**params)
        else:
            model = CatBoostRegressor(**params)
        model.fit(x, target)

        # --- Stage: Splits + LeafValues (model.json) ------------------------
        model.save_model(str(scenario_dir / "model.json"), format="json")

        # --- Stage: StagedApprox (raw approximant / logit) ------------------
        if pred_type is not None:
            staged = [
                np.asarray(p, dtype=np.float64)
                for p in model.staged_predict(x, prediction_type=pred_type)
            ]
        else:
            staged = [np.asarray(p, dtype=np.float64) for p in model.staged_predict(x)]
        staged_flat = _assert_f64(
            np.concatenate([s.ravel() for s in staged]).astype(np.float64), "staged"
        )
        np.save(scenario_dir / "staged.npy", staged_flat, allow_pickle=False)

        config = {
            "scenario": f"leaf_methods/{name}",
            "seed": SEED,
            "catboost_version": CATBOOST_VERSION,
            "thread_count": 1,
            "input_dataset": "numeric_tiny",
            "loss_function": loss,
            "leaf_estimation_method": estimator,
            "boost_from_average": bfa,
            "params": params,
            "n_rows": int(x.shape[0]),
            "n_features": int(x.shape[1]),
            "n_iterations": len(staged),
            "prediction_type": pred_type or "RawFormulaVal",
            "stages": ["Splits", "LeafValues", "StagedApprox"],
            "staged_layout": (
                "flat f64: stage 0 (n_rows), then stage 1, ... ; n_iterations "
                "stages (raw approximant / logit)"
            ),
            "leaf_formula_note": (
                "Gradient: CalcAverage(SumDer,SumWeights,scaledL2). "
                "Newton: SumDer/(-SumDer2+scaledL2) (Logloss der2=-p(1-p) makes it "
                "distinct from Gradient; RMSE der2=-1 makes Newton==Gradient). "
                "Exact: CalcOneDimensionalOptimumConstApprox -> weighted quantile "
                "(MAE alpha=0.5,delta=1e-6) of leaf residuals (target-approx). "
                "Simple: == Gradient leaf-delta (A6, CalcLeafDeltasSimple Gradient "
                "branch)."
            ),
        }
        with (scenario_dir / "config.json").open("w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2, sort_keys=True)
            fh.write("\n")


def gen_bootstrap() -> None:
    """bootstrap/{no,bayesian,bernoulli,mvs}/ — the TRAIN-04 (D-10) sampling
    oracle. Each scenario pins ONE `bootstrap_type` (+ the matching `subsample`
    or `bagging_temperature`) and every other knob at the first-slice simplified
    isolating values, so an end-to-end divergence is attributable to the sampler.

    A DEDICATED multi-block dataset (1500 objects > 1000) is synthesized and
    committed as `inputs/bootstrap_multiblock/` so the per-1000-element-block
    reseed contract (Pitfall 4: `TRestorableFastRng64(randSeed + blockIdx)` then
    `Advance(10)`, block size 1000) is actually exercised across >=2 blocks for
    Bayesian/Bernoulli (MVS uses a single 8192-element block, so it is locked
    end-to-end only — D-11: MVS internal weights are not Python-observable).

    POISSON IS DELIBERATELY ABSENT: upstream rejects `bootstrap_type=Poisson` on
    CPU (`bootstrap_options.cpp:27-33` -- "poisson bootstrap is not supported on
    CPU"), so there is NO Python-reachable CPU oracle for it. The Rust dispatch
    mirrors upstream by surfacing a `CbError` for Poisson on the CPU path; it is
    validated only by a unit test of the dispatch error, never an oracle.

    UPSTREAM DRAW SEMANTICS (transcribed from tensor_search_helpers.cpp /
    calc_score_cache.cpp / mvs.cpp for the Rust Task-2 implementation; CPU,
    SamplingFrequency=PerTree default, object sampling unit, non-pairwise loss):
      - Bootstrap runs ONCE PER TREE (greedy_tensor_search.cpp:1916), with the
        persistent LearnProgress->Rand (seeded `random_seed`) advancing across
        every iteration -- the draw stream is continuous, NOT reseeded per tree.
      - No        : SampleWeights all 1.0; SetControl all true; zero RNG draws.
      - Bayesian  : GenerateRandomWeights -- randSeed = Rand.GenRand(); per
        1000-block `r = TRestorableFastRng64(randSeed + blockIdx); r.Advance(10)`;
        per object `w = powf(-ln(r.GenRandReal1() + 1e-100), bagging_temperature)`.
        SampleWeights[i] = w (Control all true -- performRandomChoice path uses
        SetSampledControl but BernoulliSampleRate==1 so no draw).
      - Bernoulli : SampleWeights all 1.0 (Fill); the object subsample lives in
        TCalcScoreFold::Sample -> SetSampledControl, which draws SEQUENTIALLY from
        the SAME continuous Rand (NO per-block reseed): `Control[i] =
        Rand.GenRandReal1() < subsample`. Only sampled (Control true) objects feed
        the split-score histograms; leaf VALUES are computed on the full fold
        (SampleWeights==1).
      - MVS       : performRandomChoice=false; single 8192-block;
        lambda = (mean|grad|)^2 on iter 0 (CalculateMeanGradValue), else
        (mean last-iter leaf L2 norm)^2; per block threshold via CalculateThreshold
        over sqrt(lambda + der^2); per object prob = GetSingleProbability(
        sqrt(grad2+lambda), threshold); randSeed = Rand.GenRand(); per 8192-block
        `r = TRestorableFastRng64(randSeed + blockIdx); r.Advance(10)`;
        SampleWeights[i] = (1/prob) * (r.GenRandReal1() < prob). Control = weight>eps.

    staged.npy stores the per-iteration RAW approximant; model.json carries the
    learning_rate-scaled leaf_values (the end-to-end <=1e-5 parity targets)."""
    BOOTSTRAP.mkdir(parents=True, exist_ok=True)
    BOOTSTRAP_INPUT.mkdir(parents=True, exist_ok=True)

    # Dedicated multi-block dataset: 1500 objects (>= 2 blocks of 1000), 4 numeric
    # features. Deterministic from a fixed seed; committed as a frozen input.
    n_rows, n_cols = 1500, 4
    rng = np.random.default_rng(INPUT_SEED_BOOTSTRAP)
    x = _assert_f64(rng.standard_normal((n_rows, n_cols)).astype(np.float64), "bootstrap X")
    coeffs = np.array([1.5, -2.0, 0.5, 3.0], dtype=np.float64)
    y = _assert_f64(
        (x @ coeffs + 0.1 * rng.standard_normal(n_rows)).astype(np.float64),
        "bootstrap y",
    )
    np.save(BOOTSTRAP_INPUT / "X.npy", x, allow_pickle=False)
    np.save(BOOTSTRAP_INPUT / "y.npy", y, allow_pickle=False)
    with (BOOTSTRAP_INPUT / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "scenario": "inputs/bootstrap_multiblock",
                "seed": INPUT_SEED_BOOTSTRAP,
                "n_rows": n_rows,
                "n_features": n_cols,
                "target": "x @ [1.5,-2.0,0.5,3.0] + 0.1*N(0,1) (RMSE regression)",
                "note": "1500 objects (>= 2 reseed blocks of 1000) for the TRAIN-04 bootstrap oracle.",
            },
            fh,
            indent=2,
            sort_keys=True,
        )
        fh.write("\n")

    # (name, bootstrap_type, extra-params). RMSE + boost_from_average=True for all
    # (the first-slice regression path); only the sampler differs per scenario.
    scenarios = [
        ("no", "No", {}),
        ("bayesian", "Bayesian", {"bagging_temperature": 1.0}),
        ("bernoulli", "Bernoulli", {"subsample": 0.8}),
        ("mvs", "MVS", {"subsample": 0.8}),
    ]

    # Shared isolating params WITHOUT the default bootstrap (overridden per
    # scenario). boost_from_average=True (RMSE). iterations bumped to 3 so the
    # continuous RNG stream advances across multiple per-tree Bootstrap calls.
    shared = {k: v for k, v in ISOLATING_PARAMS.items() if k != "bootstrap_type"}
    shared = {**shared, "iterations": 3, "boost_from_average": True}

    for name, bt, extra in scenarios:
        scenario_dir = BOOTSTRAP / name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        params = {**shared, "bootstrap_type": bt, **extra}

        model = CatBoostRegressor(**params)
        model.fit(x, y)

        model.save_model(str(scenario_dir / "model.json"), format="json")

        staged = [np.asarray(p, dtype=np.float64) for p in model.staged_predict(x)]
        staged_flat = _assert_f64(
            np.concatenate([s.ravel() for s in staged]).astype(np.float64), "staged"
        )
        np.save(scenario_dir / "staged.npy", staged_flat, allow_pickle=False)

        predictions = _assert_f64(
            np.asarray(model.predict(x), dtype=np.float64), "predictions"
        )
        np.save(scenario_dir / "predictions.npy", predictions, allow_pickle=False)

        config = {
            "scenario": f"bootstrap/{name}",
            "seed": SEED,
            "catboost_version": CATBOOST_VERSION,
            "thread_count": 1,
            "input_dataset": "bootstrap_multiblock",
            "loss_function": "RMSE",
            "bootstrap_type": bt,
            "boost_from_average": True,
            "params": params,
            "n_rows": int(x.shape[0]),
            "n_features": int(x.shape[1]),
            "n_iterations": len(staged),
            "stages": ["Splits", "LeafValues", "StagedApprox", "Predictions"],
            "staged_layout": (
                "flat f64: stage 0 (n_rows), then stage 1, ... ; n_iterations "
                "stages (raw approximant)"
            ),
            "draw_note": (
                "CPU SamplingFrequency=PerTree, object sampling, non-pairwise. "
                "Bootstrap runs once per tree on the continuous LearnProgress->Rand "
                "(seeded random_seed). No=all 1.0/no draws. Bayesian=per-1000-block "
                "reseed (randSeed+blockIdx, Advance(10)) powf(-ln(GenRandReal1()+1e-100"
                "),temp). Bernoulli=Fill(1) + SetSampledControl GenRandReal1()<subsample "
                "sequential on the SAME Rand (no block reseed). MVS=single 8192-block "
                "threshold sampler (mvs.cpp). Poisson is rejected on CPU upstream -- no "
                "oracle exists for it."
            ),
        }
        with (scenario_dir / "config.json").open("w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2, sort_keys=True)
            fh.write("\n")


def main() -> None:
    # Load the frozen numeric_tiny input corpus.
    x = np.load(INPUTS / "numeric_tiny" / "X.npy")
    y = np.load(INPUTS / "numeric_tiny" / "y.npy")

    # RMSE regression skeleton: simplified isolating params (D-07) with
    # boost_from_average=True (Pitfall 2: RMSE boosts from the label mean).
    model = CatBoostRegressor(boost_from_average=True, **ISOLATING_PARAMS)
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
        "loss_function": "RMSE",
        "boost_from_average": True,
        "params": {**ISOLATING_PARAMS, "boost_from_average": True},
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

    # --- binclf_skeleton (Logloss training oracle, D-07/D-08) ---------------
    gen_binclf_skeleton()
    print(f"Wrote binclf_skeleton oracle fixtures under {BINCLF_SCENARIO}")

    # --- leaf_methods (TRAIN-03 four-method leaf oracle, D-09) --------------
    gen_leaf_methods()
    print(f"Wrote leaf_methods oracle fixtures under {LEAF_METHODS}")

    # --- bootstrap (TRAIN-04 sampling oracle, D-10) -------------------------
    gen_bootstrap()
    print(f"Wrote bootstrap oracle fixtures under {BOOTSTRAP}")

    # --- Wave-0 scenarios (A1-A5 resolution) --------------------------------
    gen_borders_quant()
    print(f"Wrote borders_quant oracle fixtures under {BORDERS_QUANT}")
    gen_cat_hash()
    print(f"Wrote cat_hash oracle fixtures under {CAT_HASH}")
    gen_class_weights()
    print(f"Wrote class_weights oracle fixtures under {CLASS_WEIGHTS}")


if __name__ == "__main__":
    main()
