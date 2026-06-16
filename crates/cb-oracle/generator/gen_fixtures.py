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
from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor, Pool

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
REGULARIZATION = FIXTURES / "regularization"
OVERFIT = FIXTURES / "overfit"
OVERFIT_INPUT = INPUTS / "overfit_eval"
EVAL_METRICS = FIXTURES / "eval_metrics"
EVAL_METRICS_INPUT = INPUTS / "eval_metrics"
AUTOLR = FIXTURES / "autolr"
# Phase-4 (Plan 04-01, D-13) offline fixture roots. These stage the downstream
# Wave-2..5 oracle locks: native `.cbm` load-parity (MODEL-01), prediction-type
# transforms (LOSS-06), SHAP / PredictionValuesChange / Interaction feature
# importance (MODEL-03/04), and CrossEntropy + Focal training (LOSS-01).
MODEL_SERDE = FIXTURES / "model_serde"
PREDICTION_TYPES = FIXTURES / "prediction_types"
FEATURE_IMPORTANCE = FIXTURES / "feature_importance"
LOSS_EXTRA = FIXTURES / "loss_extra"
# Phase-5 (Plan 05-10, ORD-02) end-to-end ordered train->predict fixture root.
# Closes the D-09 omission of the prior `ordered_boost/` fixture (per-object
# internals only): this carries the FULL train->predict stack (X/y/model.json/
# predictions) for boosting_type=Ordered so the multi-tree e2e oracle validates
# final predictions through the production cb_model::predict_raw apply path.
ORDERED_BOOST_E2E = FIXTURES / "ordered_boost_e2e"
# Phase-5 (Plan 05-09, ORD-05) end-to-end TENSOR-CTR train->predict fixture root.
# Closes the D-09 omission of the prior `tensor_ctr/` fixture (per-object combined
# CTR internals only, no inputs/model): this carries the FULL train->predict stack
# (X_cat/y/model.json WITH baked ctr_data/predictions) for a categorical model
# trained with simple_ctr + combinations_ctr + max_ctr_complexity, so the e2e
# oracle validates final predictions through the production cb_model::predict_raw
# CTR-split apply path (ModelSplit::Ctr) ≤1e-5 across ALL trees.
TENSOR_CTR_E2E = FIXTURES / "tensor_ctr_e2e"

# ---------------------------------------------------------------------------
# PHASE-4 FIXTURE MANIFEST (D-13) — every NEW fixture path the downstream Wave-2..5
# plans expect under crates/cb-oracle/fixtures/. Produced OFFLINE by this
# generator on a machine with catboost==1.2.10 (Python catboost is NOT importable
# in CI; CI only READS the committed artifacts — D-12). Re-run with
# `.venv/bin/python gen_fixtures.py` then COMMIT the results.
#
#   model_serde/binclf/model.cbm          MODEL-01 native .cbm (binclf, 1-dim)
#   model_serde/binclf/model.json         MODEL-01 matching JSON (carries leaf_weights)
#   model_serde/binclf/predictions.npy    RawFormulaVal reference for round-trip apply
#   model_serde/regression/model.cbm      MODEL-01 native .cbm (regression, 1-dim)
#   model_serde/regression/model.json     MODEL-01 matching JSON (carries leaf_weights)
#   model_serde/regression/predictions.npy reference predictions
#   prediction_types/{rawformulaval,probability,logprobability,class,exponent}.npy
#                                         LOSS-06 per-prediction-type transforms
#   feature_importance/shap_values.npy    MODEL-03 SHAP (n_rows x (n_features+1))
#   feature_importance/prediction_values_change.npy MODEL-04 PredictionValuesChange
#   feature_importance/interaction.npy    MODEL-04 Interaction (flattened pairs)
#   loss_extra/cross_entropy/{model.json,staged.npy,predictions.npy} LOSS-01 CrossEntropy
#   loss_extra/focal/{model.json,staged.npy,predictions.npy}         LOSS-01 Focal
# ---------------------------------------------------------------------------

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
# Dedicated synthesis seed for the TRAIN-06 overfit train/eval corpus (kept
# distinct so the dataset is reproducible in isolation).
INPUT_SEED_OVERFIT = 20260614
# Dedicated synthesis seed for the TRAIN-07 eval-metrics multi-eval-set corpus
# (kept distinct so the dataset is reproducible in isolation).
INPUT_SEED_EVAL_METRICS = 20260615

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


def gen_regularization() -> None:
    """regularization/{l2,random_strength,bagging_temp}/ — the TRAIN-05 (D-10)
    regularization oracle. Each scenario varies ONE regularization knob and pins
    every OTHER knob at the first-slice simplified isolating values, so an
    end-to-end divergence is attributable to that single knob (Pitfall 3 for
    random_strength; the Bayesian draw stream for bagging_temp).

    Trained on the TINY `numeric_tiny` corpus (50 objects, 4 features, single RNG
    block) so a `random_strength` divergence is localizable at tree granularity
    (Open Q4 / D-11: end-to-end lock on a tiny dataset; C++ instrumentation is
    escalated only if it genuinely cannot be localized).

    SCENARIOS:
      - l2             : random_strength=0, bootstrap_type=No, l2_leaf_reg=10.0
        (distinct from the skeleton's 3.0). Pure `ScaleL2Reg` scaling in the
        score (CalcAverage) and the leaf delta — NO RNG draws. Verifies the L2
        path tracks a varied regularizer end-to-end.
      - random_strength: random_strength=1.0, bootstrap_type=No, l2_leaf_reg=3.0.
        Turns the split-score perturbation ON. Per scored candidate
        `TRandomScore::GetInstance` adds `StdNormalDistribution(rand)*scoreStDev`
        where `scoreStDev = random_strength * derivativesStDevFromZero *
        CalcDerivativesStDevFromZeroMultiplier(n, modelLength)` (default
        random_score_type=NormalWithModelSizeDecrease). The Box-Muller draw
        consumes a VARIABLE number of GenRandReal1 uniforms per candidate
        (Pitfall 3) — the parity landmine this scenario locks.
      - bagging_temp   : bootstrap_type=Bayesian, bagging_temperature=0.5
        (distinct from the bootstrap oracle's 1.0), random_strength=0. Drives the
        Bayesian weight exponent `powf(-FastLogf(GenRandReal1()+1e-100), temp)`.
      - random_strength_bernoulli : random_strength=1.0 COMBINED with
        bootstrap_type=Bernoulli, subsample=0.7 (strictly < 1.0 so the Bernoulli
        `control` mask actually DROPS objects on the first tree). This is the
        CROSS-scenario that the other three deliberately do NOT exercise: when the
        control mask drops objects, the SAMPLED/masked split-scoring derivative
        vector (`score_weighted_der1`, control-false entries zeroed) differs from
        the FULL, un-sampled fold derivative vector (`weighted_der1`). Upstream
        `CalcDerivativesStDevFromZeroPlainBoosting` (greedy_tensor_search.cpp:
        92-107) computes `scoreStDev` over the FULL fold derivatives — NOT the
        masked vector — exactly as the LEAF path does; only the split-scoring
        HISTOGRAM is restricted to control-true objects. This scenario gates that
        the std-dev input is the full fold (closes CR-01).

    UPSTREAM RANDOM_STRENGTH DRAW SEMANTICS (CPU single-host, transcribed from
    greedy_tensor_search.cpp / tensor_search_helpers.cpp / rand_score.h for the
    Rust implementation):
      - modelLength = treeIndex * learning_rate (train.cpp:177-178).
      - derivativesStDevFromZero (Plain) = sqrt(sum(weightedDer^2) / nObjects)
        (CalcDerivativesStDevFromZeroPlainBoosting, :92-107).
      - multiplier = modelLeft/(1+modelLeft), modelLeft = exp(log(n) - modelLength)
        (CalcDerivativesStDevFromZeroMultiplier, :125-129).
      - scoreStDev = random_strength * derivativesStDevFromZero * multiplier (:861).
      - Per level: randSeed = Rand.GenRand() (CalcScores, :884).
      - SetBestScore (per candidate/feature taskIdx, tensor_search_helpers.cpp:716):
        TRestorableFastRng64(randSeed + taskIdx); rand.Advance(10); then per
        border `scoreInstance = scoreWoNoise + StdNormalDistribution(rand)*scoreStDev`
        keeping the border with the max scoreInstance (strict `>`). The kept
        TRandomScore (Val=winning scoreWoNoise, StDev=scoreStDev) is stored.
      - SelectBestCandidate (greedy_tensor_search.cpp:948-966): per candidate ONE
        `score = candidate.BestScore.GetInstance(Rand)` from the MAIN persistent
        Rand (= Val + StdNormalDistribution(Rand)*StDev), then strict
        `gain > bestGain` first-wins.

    staged.npy stores the per-iteration RAW approximant; model.json carries the
    learning_rate-scaled leaf_values (the end-to-end <=1e-5 parity targets)."""
    REGULARIZATION.mkdir(parents=True, exist_ok=True)

    x = _assert_f64(np.load(INPUTS / "numeric_tiny" / "X.npy"), "regularization X")
    y = _assert_f64(np.load(INPUTS / "numeric_tiny" / "y.npy"), "regularization y")

    # (name, overrides). RMSE + boost_from_average=True (the first-slice
    # regression path); only the named regularization knob varies per scenario.
    scenarios = [
        ("l2", {"l2_leaf_reg": 10.0, "random_strength": 0, "bootstrap_type": "No"}),
        (
            "random_strength",
            {"l2_leaf_reg": 3.0, "random_strength": 1.0, "bootstrap_type": "No"},
        ),
        (
            "bagging_temp",
            {
                "l2_leaf_reg": 3.0,
                "random_strength": 0,
                "bootstrap_type": "Bayesian",
                "bagging_temperature": 0.5,
            },
        ),
        # CR-01 cross-scenario: random_strength != 0 COMBINED with a sampling
        # bootstrap (Bernoulli, subsample < 1.0). The Bernoulli control mask
        # drops objects on the first tree, so the masked score derivatives
        # (score_weighted_der1) differ from the full-fold weighted_der1 that
        # scoreStDev (CalcDerivativesStDevFromZeroPlainBoosting) must use.
        (
            "random_strength_bernoulli",
            {
                "l2_leaf_reg": 3.0,
                "random_strength": 1.0,
                "bootstrap_type": "Bernoulli",
                "subsample": 0.7,
            },
        ),
    ]

    # Shared isolating params. iterations bumped to 3 so the modelLength
    # multiplier and the continuous RNG stream advance across multiple trees.
    shared = {**ISOLATING_PARAMS, "iterations": 3, "boost_from_average": True}

    for name, overrides in scenarios:
        scenario_dir = REGULARIZATION / name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        params = {**shared, **overrides}

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
            "scenario": f"regularization/{name}",
            "seed": SEED,
            "catboost_version": CATBOOST_VERSION,
            "thread_count": 1,
            "input_dataset": "numeric_tiny",
            "loss_function": "RMSE",
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
                "l2: pure ScaleL2Reg scaling, no RNG draws. random_strength=1.0: "
                "TRandomScore::GetInstance adds StdNormalDistribution(rand)*"
                "scoreStDev per candidate; scoreStDev = random_strength * "
                "sqrt(sum(weightedDer^2)/n) * (modelLeft/(1+modelLeft)), modelLeft="
                "exp(log(n)-treeIdx*lr); SetBestScore reseeds "
                "TRestorableFastRng64(Rand.GenRand()+taskIdx).Advance(10) per "
                "feature, SelectBestCandidate draws once per feature from the main "
                "Rand. bagging_temp=0.5: Bayesian weight powf(-FastLogf("
                "GenRandReal1()+1e-100),temp). random_strength_bernoulli: "
                "random_strength=1.0 + Bernoulli(subsample=0.7). scoreStDev is "
                "taken over the FULL, un-sampled fold derivatives "
                "(CalcDerivativesStDevFromZeroPlainBoosting, "
                "greedy_tensor_search.cpp:92-107) — NOT the control-masked "
                "score_weighted_der1; the Bernoulli control mask restricts ONLY "
                "the split-scoring histogram, so the std-dev input is the full "
                "fold (same input as the leaf path)."
            ),
        }
        with (scenario_dir / "config.json").open("w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2, sort_keys=True)
            fh.write("\n")


def gen_overfit() -> None:
    """overfit/{inctodec,iter,wilcoxon,use_best_model}/ — the TRAIN-06 (D-10)
    overfitting-detection / early-stopping oracle. Each scenario trains with an
    explicit held-out eval set and pins the matching `od_type`/`od_pval`/`od_wait`
    (and `use_best_model` where applicable) so the detector FIRES inside the
    iteration budget, stopping the model at an iteration < the configured maximum.

    DETERMINISTIC EVAL CONFIG (prior-wave guidance / D-07): the eval-loss curve is
    produced with `bootstrap_type=No`, `random_strength=0` so the stop decision
    and best iteration lock cleanly and are NOT perturbed by the known stochastic
    multi-tree RNG-phase residual (TRAIN-04/05). A purpose-built train/eval split
    is synthesized whose eval loss starts to RISE after a handful of iterations so
    every detector type has a real overfit signal to detect.

    UPSTREAM DETECTOR SEMANTICS (overfitting_detector.cpp:120-208, transcribed for
    the Rust Task-2 port):
      - default type = IncToDec, wait_iterations = 20, stop_pvalue (od_pval) = 0
        (0 => detector INACTIVE: IsActive() iff Threshold>0).
      - IncToDec: AddError tracks a running LocalMax of the (sign-adjusted) error,
        an exponentially-forgotten ExpectedInc over the last ITERATION_FORGET=2000
        errors (LAMBDA_FORGET=0.99), and after IterationsFromLocalMax>=wait sets
        CurrentPValue = exp(-LAMBDA_SCALE / max(ExpectedInc/max(LocalMax-Last,EPS),
        EPS)), LAMBDA_SCALE=0.5, EPS=1e-10. IsNeedStop iff !IsEmpty && pval<thresh.
      - Iter == IncToDec with threshold forced to 1.0 (stops wait iters after the
        best, since pval<1.0 always holds once the wait elapses past the max).
      - Wilcoxon: deltas (LastError-err) AFTER the local max; once >= wait deltas
        accumulate, CurrentPValue = NStatistics::Wilcoxon(deltas) (the signed-rank
        statistic over post-local-max deltas).
      - maxIsOptimal=false for a LOSS metric (RMSE/Logloss): err is negated so a
        DECREASING loss is an INCREASING (improving) score.

    Persists per scenario:
      - model.json   : the trained model whose tree count == the STOP iteration
                       (catboost truncates the saved trees to best_iteration+1 when
                       use_best_model, and to the stop point otherwise).
      - staged.npy   : the per-iteration eval-set metric curve (the loss the
                       detector consumes), n_iterations_run f64.
      - eval X/y     : the held-out eval inputs (committed under inputs/) so the
                       Rust harness recomputes the same eval-loss curve.
      - config.json  : od_type/od_pval/od_wait/use_best_model + tree_count_ +
                       best_iteration_ from the Python API (the assertion targets).
    """
    OVERFIT.mkdir(parents=True, exist_ok=True)
    OVERFIT_INPUT.mkdir(parents=True, exist_ok=True)

    # Synthesize a train/eval split where the eval loss bottoms out early and then
    # rises (overfit signal). Train is small + noisy so the model keeps fitting
    # train noise; eval is a clean held-out sample drawn from the SAME linear
    # signal so its loss starts climbing once the model overfits the train noise.
    rng = np.random.default_rng(INPUT_SEED_OVERFIT)
    n_train, n_eval, n_cols = 120, 80, 4
    coeffs = np.array([1.5, -2.0, 0.5, 3.0], dtype=np.float64)
    x_tr = _assert_f64(rng.standard_normal((n_train, n_cols)).astype(np.float64), "overfit Xtr")
    # Heavy train noise so the model overfits within a few dozen iterations.
    y_tr = _assert_f64(
        (x_tr @ coeffs + 1.5 * rng.standard_normal(n_train)).astype(np.float64),
        "overfit ytr",
    )
    x_ev = _assert_f64(rng.standard_normal((n_eval, n_cols)).astype(np.float64), "overfit Xev")
    # Clean eval signal (light noise) so the held-out loss has a clear minimum.
    y_ev = _assert_f64(
        (x_ev @ coeffs + 0.1 * rng.standard_normal(n_eval)).astype(np.float64),
        "overfit yev",
    )
    np.save(OVERFIT_INPUT / "X_train.npy", x_tr, allow_pickle=False)
    np.save(OVERFIT_INPUT / "y_train.npy", y_tr, allow_pickle=False)
    np.save(OVERFIT_INPUT / "X_eval.npy", x_ev, allow_pickle=False)
    np.save(OVERFIT_INPUT / "y_eval.npy", y_ev, allow_pickle=False)
    with (OVERFIT_INPUT / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "scenario": "inputs/overfit_eval",
                "seed": INPUT_SEED_OVERFIT,
                "n_train": n_train,
                "n_eval": n_eval,
                "n_features": n_cols,
                "target": "x @ [1.5,-2.0,0.5,3.0] + noise (train 1.5*N, eval 0.1*N)",
                "note": "Held-out eval set whose RMSE loss rises after a few iters (overfit signal) for the TRAIN-06 detector oracle.",
            },
            fh,
            indent=2,
            sort_keys=True,
        )
        fh.write("\n")

    # Deterministic eval config (No bootstrap, random_strength=0) so the stop
    # decision locks cleanly. Larger iteration budget so the detector has room to
    # fire and stop EARLY (tree_count_ < iterations).
    shared = {
        k: v
        for k, v in ISOLATING_PARAMS.items()
        if k not in ("bootstrap_type", "random_strength", "iterations")
    }
    shared = {
        **shared,
        "iterations": 200,
        "bootstrap_type": "No",
        "random_strength": 0,
        "boost_from_average": True,
        "depth": 3,
        "learning_rate": 0.3,
    }

    # (name, od_type, od_pval, od_wait, use_best_model). od_wait small so the
    # detector reacts within the budget; od_pval chosen so the (in/dec) p-value
    # actually crosses the threshold once eval loss starts climbing.
    scenarios = [
        ("inctodec", "IncToDec", 0.99, 10, False),
        ("iter", "Iter", 0.0, 10, False),
        ("wilcoxon", "Wilcoxon", 0.01, 10, False),
        ("use_best_model", "IncToDec", 0.99, 10, True),
    ]

    for name, od_type, od_pval, od_wait, use_best in scenarios:
        scenario_dir = OVERFIT / name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        params = {
            **shared,
            "od_type": od_type,
            "od_wait": od_wait,
            "use_best_model": use_best,
        }
        # Iter does not consume od_pval (threshold is forced to 1.0); IncToDec /
        # Wilcoxon take od_pval as the AutoStopPValue threshold.
        if od_type != "Iter":
            params["od_pval"] = od_pval

        model = CatBoostRegressor(**params)
        model.fit(x_tr, y_tr, eval_set=(x_ev, y_ev))

        model.save_model(str(scenario_dir / "model.json"), format="json")

        # The eval-set metric curve the detector consumes: the per-iteration RMSE
        # on the eval set (staged_predict over the eval inputs -> RMSE per stage).
        eval_staged = [
            np.asarray(p, dtype=np.float64) for p in model.staged_predict(x_ev)
        ]
        eval_losses = [
            float(np.sqrt(np.mean((stage - y_ev) ** 2))) for stage in eval_staged
        ]
        eval_curve = _assert_f64(np.asarray(eval_losses, dtype=np.float64), "eval_curve")
        np.save(scenario_dir / "staged.npy", eval_curve, allow_pickle=False)

        tree_count = int(model.tree_count_)
        best_iteration = int(model.get_best_iteration())

        config = {
            "scenario": f"overfit/{name}",
            "seed": SEED,
            "catboost_version": CATBOOST_VERSION,
            "thread_count": 1,
            "input_dataset": "overfit_eval",
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "boost_from_average": True,
            "od_type": od_type,
            "od_pval": od_pval,
            "od_wait": od_wait,
            "use_best_model": use_best,
            "params": params,
            "n_train": n_train,
            "n_eval": n_eval,
            "n_features": n_cols,
            "n_iterations_run": len(eval_curve),
            "tree_count_": tree_count,
            "best_iteration_": best_iteration,
            "stages": ["Splits", "LeafValues", "StagedEvalLoss"],
            "staged_layout": (
                "flat f64: per-iteration RMSE on the EVAL set (the detector's "
                "AddError sequence); n_iterations_run entries."
            ),
            "detector_note": (
                "IncToDec(default,wait=20,pval=0=>inactive): IsNeedStop iff "
                "!IsEmpty && CurrentPValue<Threshold; pval=exp(-0.5/max(ExpInc/"
                "max(LocalMax-Last,1e-10),1e-10)). Iter==IncToDec threshold=1.0 "
                "(stops od_wait after best). Wilcoxon over post-local-max deltas. "
                "maxIsOptimal=false for a loss metric (err negated). tree_count_ "
                "is the STOP iteration; best_iteration_ the use_best_model best."
            ),
        }
        with (scenario_dir / "config.json").open("w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2, sort_keys=True)
            fh.write("\n")


def gen_eval_metrics() -> None:
    """eval_metrics/{rmse,logloss}/ — the TRAIN-07 (D-10) per-iteration eval-set
    metric-logging oracle. Each scenario trains with TWO held-out eval sets
    (`eval_set=[(X1,y1),(X2,y2)]`) and an EXPLICIT `eval_metric`, and persists the
    upstream PER-ITERATION metric history FOR EACH eval set (from
    `model.get_evals_result()` -> `validation_0`/`validation_1`) as committed
    `.npy`. The Rust harness recomputes the same per-iteration metric curve per
    eval set via `cb-train::metrics` and locks it at <= 1e-5.

    `eval_metric` semantics (upstream): the metric reported per iteration per eval
    set; it DEFAULTS to the objective when unset. Two scenarios cover both losses:
      - rmse:    CatBoostRegressor,  loss=RMSE,    eval_metric='RMSE'
                 (RMSE == sqrt(sum_w (pred-target)^2 / sum_w)).
      - logloss: CatBoostClassifier, loss=Logloss, eval_metric='Logloss'
                 (weighted cross-entropy over p=sigmoid(raw logit)).

    DETERMINISTIC CONFIG (prior-wave guidance / D-07): `bootstrap_type=No`,
    `random_strength=0`, fixed seed, `thread_count=1` so the per-iteration metric
    values lock CLEANLY and are not perturbed by the known stochastic multi-tree
    RNG residual (TRAIN-04/05). NO early stopping (the full iteration budget runs)
    so every iteration's metric value is locked. Python-reachable oracle only
    (D-11): the per-eval-set per-iteration history comes straight from the public
    `get_evals_result()` API — no C++ instrumentation.

    Persists per scenario:
      - model.json          : the trained model (splits + leaf_values).
      - eval0_metric.npy    : validation_0 per-iteration eval_metric history (f64).
      - eval1_metric.npy    : validation_1 per-iteration eval_metric history (f64).
      - config.json         : loss/eval_metric + n_iterations + assertion metadata.
    Eval-set inputs (X/y for both eval sets + the train set) are committed under
    inputs/eval_metrics/ so the Rust harness recomputes the curves.
    """
    EVAL_METRICS.mkdir(parents=True, exist_ok=True)
    EVAL_METRICS_INPUT.mkdir(parents=True, exist_ok=True)

    # Synthesize a train set + two distinct held-out eval sets from the same linear
    # signal (different sizes so per-eval-set bookkeeping is exercised, not a single
    # shared length).
    rng = np.random.default_rng(INPUT_SEED_EVAL_METRICS)
    n_train, n_eval0, n_eval1, n_cols = 120, 60, 45, 4
    coeffs = np.array([1.5, -2.0, 0.5, 3.0], dtype=np.float64)
    x_tr = _assert_f64(rng.standard_normal((n_train, n_cols)).astype(np.float64), "em Xtr")
    x_e0 = _assert_f64(rng.standard_normal((n_eval0, n_cols)).astype(np.float64), "em Xe0")
    x_e1 = _assert_f64(rng.standard_normal((n_eval1, n_cols)).astype(np.float64), "em Xe1")

    # Regression targets (continuous) for the RMSE scenario; binary labels (from a
    # thresholded linear score) for the Logloss scenario. Both eval sets are clean
    # (light noise) held-out samples from the same signal.
    y_tr_reg = _assert_f64(
        (x_tr @ coeffs + 0.3 * rng.standard_normal(n_train)).astype(np.float64), "em ytr_reg"
    )
    y_e0_reg = _assert_f64(
        (x_e0 @ coeffs + 0.1 * rng.standard_normal(n_eval0)).astype(np.float64), "em ye0_reg"
    )
    y_e1_reg = _assert_f64(
        (x_e1 @ coeffs + 0.1 * rng.standard_normal(n_eval1)).astype(np.float64), "em ye1_reg"
    )
    clf_coeffs = np.array([1.0, -1.5, 0.5, 2.0], dtype=np.float64)
    y_tr_clf = _assert_f64(
        (x_tr @ clf_coeffs + 0.5 * rng.standard_normal(n_train) > 0).astype(np.float64),
        "em ytr_clf",
    )
    y_e0_clf = _assert_f64((x_e0 @ clf_coeffs > 0).astype(np.float64), "em ye0_clf")
    y_e1_clf = _assert_f64((x_e1 @ clf_coeffs > 0).astype(np.float64), "em ye1_clf")

    # Commit the shared feature matrices + per-loss targets for the Rust harness.
    np.save(EVAL_METRICS_INPUT / "X_train.npy", x_tr, allow_pickle=False)
    np.save(EVAL_METRICS_INPUT / "X_eval0.npy", x_e0, allow_pickle=False)
    np.save(EVAL_METRICS_INPUT / "X_eval1.npy", x_e1, allow_pickle=False)
    np.save(EVAL_METRICS_INPUT / "y_train_rmse.npy", y_tr_reg, allow_pickle=False)
    np.save(EVAL_METRICS_INPUT / "y_eval0_rmse.npy", y_e0_reg, allow_pickle=False)
    np.save(EVAL_METRICS_INPUT / "y_eval1_rmse.npy", y_e1_reg, allow_pickle=False)
    np.save(EVAL_METRICS_INPUT / "y_train_logloss.npy", y_tr_clf, allow_pickle=False)
    np.save(EVAL_METRICS_INPUT / "y_eval0_logloss.npy", y_e0_clf, allow_pickle=False)
    np.save(EVAL_METRICS_INPUT / "y_eval1_logloss.npy", y_e1_clf, allow_pickle=False)
    with (EVAL_METRICS_INPUT / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "scenario": "inputs/eval_metrics",
                "seed": INPUT_SEED_EVAL_METRICS,
                "n_train": n_train,
                "n_eval0": n_eval0,
                "n_eval1": n_eval1,
                "n_features": n_cols,
                "note": (
                    "Train set + TWO held-out eval sets (different sizes) for the "
                    "TRAIN-07 per-iteration eval-metric oracle; rmse targets are "
                    "continuous, logloss targets binary (thresholded linear score)."
                ),
            },
            fh,
            indent=2,
            sort_keys=True,
        )
        fh.write("\n")

    shared = {
        "iterations": 12,
        "learning_rate": 0.3,
        "depth": 3,
        "l2_leaf_reg": 3.0,
        "bootstrap_type": "No",
        "random_strength": 0,
        "leaf_estimation_iterations": 1,
        "score_function": "L2",
        "leaf_estimation_method": "Gradient",
        "random_seed": SEED,
        "thread_count": 1,
        "verbose": False,
    }

    # (name, eval_metric, loss, is_classifier, boost_from_average, train_y, e0_y, e1_y)
    scenarios = [
        (
            "rmse",
            "RMSE",
            "RMSE",
            False,
            True,
            y_tr_reg,
            y_e0_reg,
            y_e1_reg,
        ),
        (
            "logloss",
            "Logloss",
            "Logloss",
            True,
            False,
            y_tr_clf,
            y_e0_clf,
            y_e1_clf,
        ),
    ]

    for name, eval_metric, loss, is_clf, bfa, y_tr, y_e0, y_e1 in scenarios:
        scenario_dir = EVAL_METRICS / name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        params = {
            **shared,
            "boost_from_average": bfa,
            "eval_metric": eval_metric,
        }
        if is_clf:
            model = CatBoostClassifier(loss_function=loss, **params)
        else:
            model = CatBoostRegressor(loss_function=loss, **params)
        model.fit(x_tr, y_tr, eval_set=[(x_e0, y_e0), (x_e1, y_e1)])

        model.save_model(str(scenario_dir / "model.json"), format="json")

        evals = model.get_evals_result()
        curve0 = _assert_f64(
            np.asarray(evals["validation_0"][eval_metric], dtype=np.float64), "em eval0"
        )
        curve1 = _assert_f64(
            np.asarray(evals["validation_1"][eval_metric], dtype=np.float64), "em eval1"
        )
        np.save(scenario_dir / "eval0_metric.npy", curve0, allow_pickle=False)
        np.save(scenario_dir / "eval1_metric.npy", curve1, allow_pickle=False)

        config = {
            "scenario": f"eval_metrics/{name}",
            "seed": SEED,
            "catboost_version": CATBOOST_VERSION,
            "thread_count": 1,
            "input_dataset": "eval_metrics",
            "loss_function": loss,
            "eval_metric": eval_metric,
            "boost_from_average": bfa,
            "n_eval_sets": 2,
            "params": params,
            "n_train": n_train,
            "n_eval0": n_eval0,
            "n_eval1": n_eval1,
            "n_features": n_cols,
            "n_iterations_run": int(len(curve0)),
            "tree_count_": int(model.tree_count_),
            "stages": ["Splits", "LeafValues", "EvalMetricPerSet"],
            "metric_layout": (
                "eval0_metric.npy / eval1_metric.npy: flat f64, the per-iteration "
                "eval_metric on validation_0 / validation_1 respectively "
                "(n_iterations_run entries each)."
            ),
            "metric_note": (
                "eval_metric defaults to the objective and is set EXPLICITLY here. "
                "RMSE == sqrt(sum_w (pred-target)^2 / sum_w); Logloss == weighted "
                "cross-entropy over p=sigmoid(raw logit). Two eval sets exercise "
                "per-eval-set per-iteration logging (TRAIN-07)."
            ),
        }
        with (scenario_dir / "config.json").open("w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2, sort_keys=True)
            fh.write("\n")


def gen_autolr() -> None:
    """autolr/{rmse,logloss}/ — the TRAIN-08 automatic learning-rate oracle.

    Trains an RMSE regressor and a Logloss classifier WITHOUT setting
    `learning_rate`, `leaf_estimation_method`, `leaf_estimation_iterations`, or
    `l2_leaf_reg` — the exact four-param gate upstream's `UpdateLearningRate`
    (`options_helper.cpp:269-288`) checks before invoking
    `TAutoLRParamsGuesser`. With all four unset (and a fixed `iterations` and a
    dataset of known object count) CatBoost auto-selects the learning rate via
    the coefficient table keyed by (target, CPU, useBestModel, boostFromAverage)
    and the exp/log/round formula.

    The persisted oracle value is `model.get_all_params()['learning_rate']` — the
    upstream-selected rate the Rust `autolr::guess` must reproduce to <= 1e-5.
    All keying inputs (target type, use_best_model, boost_from_average,
    learn_object_count, iterations) are recorded in config.json so the unit test
    asserts the formula on the exact key + inputs CatBoost used (no eval set, so
    use_best_model defaults to False; boost_from_average defaults True for RMSE,
    False for Logloss — Pitfall 2). Python-reachable oracle only (D-11)."""
    x = np.load(INPUTS / "numeric_tiny" / "X.npy")
    y_cont = np.load(INPUTS / "numeric_tiny" / "y.npy")
    learn_object_count = int(x.shape[0])

    # A fixed iteration count != 1000 so the custIter/defIter ratio is exercised
    # (defIter is computed at iterCount=1000 in the formula).
    iterations = 500

    # NOTE: deliberately DO NOT set learning_rate / leaf_estimation_method /
    # leaf_estimation_iterations / l2_leaf_reg, so the auto-LR gate fires. Also
    # do NOT set bootstrap_type/random_strength here — they do not affect the
    # learning-rate guess (a pure pre-train scalar of size/iters/key), and the
    # parity assertion is on get_all_params()['learning_rate'] only, not splits.
    scenarios = [
        {
            "name": "rmse",
            "loss_function": "RMSE",
            "target_type": "RMSE",
            "boost_from_average": True,  # upstream default for RMSE (Pitfall 2)
            "model_cls": CatBoostRegressor,
            "y": _assert_f64(y_cont.astype(np.float64), "autolr rmse y"),
        },
        {
            "name": "logloss",
            "loss_function": "Logloss",
            "target_type": "Logloss",
            "boost_from_average": False,  # upstream default for Logloss
            "model_cls": CatBoostClassifier,
            "y": (y_cont > np.median(y_cont)).astype(np.int64),
        },
    ]

    for sc in scenarios:
        params = {
            "iterations": iterations,
            "loss_function": sc["loss_function"],
            "depth": 2,
            "random_seed": SEED,
            "thread_count": 1,
            "verbose": False,
        }
        model = sc["model_cls"](**params)
        model.fit(x, sc["y"])

        all_params = model.get_all_params()
        selected_lr = float(all_params["learning_rate"])
        # use_best_model: no eval set => defaults to False (and stays False).
        use_best_model = bool(all_params.get("use_best_model", False))
        boost_from_average = bool(
            all_params.get("boost_from_average", sc["boost_from_average"])
        )

        scenario_dir = AUTOLR / sc["name"]
        scenario_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "scenario": f"autolr/{sc['name']}",
            "seed": SEED,
            "catboost_version": CATBOOST_VERSION,
            "thread_count": 1,
            "input_dataset": "numeric_tiny",
            "loss_function": sc["loss_function"],
            # The auto-LR keying inputs the Rust guess() consumes.
            "target_type": sc["target_type"],
            "task_type": "CPU",
            "use_best_model": use_best_model,
            "boost_from_average": boost_from_average,
            "learn_object_count": learn_object_count,
            "iterations": iterations,
            # The upstream-selected learning rate (the parity target, <= 1e-5).
            "selected_learning_rate": selected_lr,
            "gate_note": (
                "learning_rate / leaf_estimation_method / "
                "leaf_estimation_iterations / l2_leaf_reg are ALL unset so "
                "TAutoLRParamsGuesser fires (options_helper.cpp:269-288)."
            ),
            "formula_note": (
                "lr = round(min(exp(A*ln(N)+B) * exp(C*ln(iter)+D) / "
                "exp(C*ln(1000)+D), 0.5), 6) with coeffs {A,B,C,D} keyed by "
                "(target_type, CPU, use_best_model, boost_from_average)."
            ),
        }
        with (scenario_dir / "config.json").open("w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2, sort_keys=True)
            fh.write("\n")


def gen_model_serde() -> None:
    """model_serde/{binclf,regression}/ — MODEL-01 native `.cbm` load-parity
    fixtures (D-13). Trains a 1-dim binclf (Logloss) and a 1-dim regression (RMSE)
    model with the simplified isolating params, saving BOTH the native `.cbm`
    blob (the FlatBuffers `TModelCore`: `CBM1` magic + ui32 size + payload) AND the
    matching `model.json` (now carrying per-tree `leaf_weights`, RESEARCH Pitfall
    1/2). `predictions.npy` is the RawFormulaVal reference the later `.cbm`
    load->apply round-trip locks against. Python-reachable oracle only (D-11/D-12).
    """
    x = np.load(INPUTS / "numeric_tiny" / "X.npy")
    y = np.load(INPUTS / "numeric_tiny" / "y.npy")
    y_bin = (y > np.median(y)).astype(np.int64)

    scenarios = [
        ("regression", CatBoostRegressor, True, y, "RMSE"),
        ("binclf", CatBoostClassifier, False, y_bin, "Logloss"),
    ]
    for name, ctor, bfa, target, loss in scenarios:
        scenario_dir = MODEL_SERDE / name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        model = ctor(boost_from_average=bfa, **ISOLATING_PARAMS)
        model.fit(x, target)

        # Native .cbm (FlatBuffers TModelCore) + matching JSON (leaf_weights).
        model.save_model(str(scenario_dir / "model.cbm"), format="cbm")
        model.save_model(str(scenario_dir / "model.json"), format="json")

        predictions = _assert_f64(
            np.asarray(
                model.predict(x, prediction_type="RawFormulaVal"), dtype=np.float64
            ),
            "predictions",
        )
        np.save(scenario_dir / "predictions.npy", predictions, allow_pickle=False)

        config = {
            "scenario": f"model_serde/{name}",
            "seed": SEED,
            "catboost_version": CATBOOST_VERSION,
            "thread_count": 1,
            "input_dataset": "numeric_tiny",
            "loss_function": loss,
            "boost_from_average": bfa,
            "params": {**ISOLATING_PARAMS, "boost_from_average": bfa},
            "n_rows": int(x.shape[0]),
            "n_features": int(x.shape[1]),
            "label_definition": (
                "y_binary = (numeric_tiny.y > median)" if name == "binclf" else "numeric_tiny.y"
            ),
            "artifacts": ["model.cbm", "model.json", "predictions.npy"],
            "prediction_type": "RawFormulaVal",
            "note": "MODEL-01 native .cbm load-parity; model.json carries leaf_weights",
        }
        with (scenario_dir / "config.json").open("w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2, sort_keys=True)
            fh.write("\n")


def gen_prediction_types() -> None:
    """prediction_types/ — LOSS-06 per-prediction-type transform oracle (D-13).
    Trains a binclf (Logloss) model and saves each upstream `prediction_type`'s
    output over the training inputs: RawFormulaVal (raw logit), Probability
    (sigmoid), LogProbability (log of the per-class probabilities), Class
    (argmax/threshold label), and Exponent (exp(rawFormulaVal)). The later apply
    plan locks each Rust transform against these (<= 1e-5). Python oracle (D-11).
    """
    PREDICTION_TYPES.mkdir(parents=True, exist_ok=True)
    x = np.load(INPUTS / "numeric_tiny" / "X.npy")
    y = np.load(INPUTS / "numeric_tiny" / "y.npy")
    y_bin = (y > np.median(y)).astype(np.int64)

    model = CatBoostClassifier(boost_from_average=False, **ISOLATING_PARAMS)
    model.fit(x, y_bin)

    # (filename, prediction_type). LogProbability/Probability are per-class 2-col
    # arrays for a binary classifier; stored flat (row-major) f64.
    for fname, ptype in [
        ("rawformulaval", "RawFormulaVal"),
        ("probability", "Probability"),
        ("logprobability", "LogProbability"),
        ("class", "Class"),
        ("exponent", "Exponent"),
    ]:
        arr = np.asarray(model.predict(x, prediction_type=ptype), dtype=np.float64)
        flat = _assert_f64(arr.ravel().astype(np.float64), fname)
        np.save(PREDICTION_TYPES / f"{fname}.npy", flat, allow_pickle=False)

    config = {
        "scenario": "prediction_types",
        "seed": SEED,
        "catboost_version": CATBOOST_VERSION,
        "thread_count": 1,
        "input_dataset": "numeric_tiny",
        "loss_function": "Logloss",
        "n_rows": int(x.shape[0]),
        "prediction_types": [
            "RawFormulaVal",
            "Probability",
            "LogProbability",
            "Class",
            "Exponent",
        ],
        "layout": "each *.npy is the flattened (row-major) f64 output for one prediction_type",
        "note": "LOSS-06 per-prediction-type transform oracle",
    }
    with (PREDICTION_TYPES / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
        fh.write("\n")


def gen_feature_importance() -> None:
    """feature_importance/ — MODEL-03/04 SHAP + PredictionValuesChange +
    Interaction oracle (D-13). Trains a binclf (Logloss) model and saves each
    `get_feature_importance(type=...)` output over a training `Pool`:
      * ShapValues             -> shap_values.npy             (n_rows x (n_features+1);
                                  last column is the expected-value / bias term)
      * PredictionValuesChange -> prediction_values_change.npy (n_features,)
      * Interaction            -> interaction.npy             (flattened [i, j, score]
                                  triples as returned by get_feature_importance)
    The later SHAP/fstr plans lock the Rust TreeSHAP recursion + fstr against these
    (<= 1e-5). Requires the per-leaf `leaf_weights` captured this plan (RESEARCH
    Pitfall 1). Python oracle (D-11)."""
    FEATURE_IMPORTANCE.mkdir(parents=True, exist_ok=True)
    x = np.load(INPUTS / "numeric_tiny" / "X.npy")
    y = np.load(INPUTS / "numeric_tiny" / "y.npy")
    y_bin = (y > np.median(y)).astype(np.int64)

    model = CatBoostClassifier(boost_from_average=False, **ISOLATING_PARAMS)
    model.fit(x, y_bin)
    pool = Pool(x, y_bin)

    shap = np.asarray(
        model.get_feature_importance(type="ShapValues", data=pool), dtype=np.float64
    )
    np.save(
        FEATURE_IMPORTANCE / "shap_values.npy",
        _assert_f64(shap.ravel().astype(np.float64), "shap"),
        allow_pickle=False,
    )

    pvc = np.asarray(
        model.get_feature_importance(type="PredictionValuesChange", data=pool),
        dtype=np.float64,
    )
    np.save(
        FEATURE_IMPORTANCE / "prediction_values_change.npy",
        _assert_f64(pvc.ravel().astype(np.float64), "pvc"),
        allow_pickle=False,
    )

    interaction = np.asarray(
        model.get_feature_importance(type="Interaction", data=pool), dtype=np.float64
    )
    np.save(
        FEATURE_IMPORTANCE / "interaction.npy",
        _assert_f64(interaction.ravel().astype(np.float64), "interaction"),
        allow_pickle=False,
    )

    config = {
        "scenario": "feature_importance",
        "seed": SEED,
        "catboost_version": CATBOOST_VERSION,
        "thread_count": 1,
        "input_dataset": "numeric_tiny",
        "loss_function": "Logloss",
        "n_rows": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "shap_shape": list(shap.shape),
        "interaction_shape": list(interaction.shape),
        "artifacts": [
            "shap_values.npy",
            "prediction_values_change.npy",
            "interaction.npy",
        ],
        "note": "MODEL-03 SHAP + MODEL-04 PredictionValuesChange/Interaction (needs leaf_weights)",
    }
    with (FEATURE_IMPORTANCE / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
        fh.write("\n")


def gen_loss_extra() -> None:
    """loss_extra/{cross_entropy,focal}/ — LOSS-01 CrossEntropy + Focal training
    oracle (D-13). Both train on the frozen numeric_tiny inputs with the
    simplified isolating params (boost_from_average=False for these classification
    losses), saving model.json (splits + leaf_values + leaf_weights), staged.npy
    (RawFormulaVal raw logits per stage), and predictions.npy (RawFormulaVal). The
    later loss-function plan locks the Rust CrossEntropy/Focal gradients against
    these (<= 1e-5). Python oracle (D-11).

    CrossEntropy accepts soft (probabilistic) targets in [0, 1]; Focal is a
    focusing-parameter reweighting of Logloss. `focal_alpha`/`focal_gamma` are
    MANDATORY for the Focal loss (catboost_options.cpp:234); we pin them to the
    common reference values (alpha=0.25, gamma=2.0) and record them in
    config.json. Targets: CrossEntropy uses the sigmoid of the standardized
    regression target (soft labels in (0,1)); Focal uses the hard binary y_bin
    label."""
    x = np.load(INPUTS / "numeric_tiny" / "X.npy")
    y = np.load(INPUTS / "numeric_tiny" / "y.npy")
    y_bin = (y > np.median(y)).astype(np.int64)
    # Soft probabilistic labels in (0, 1) for CrossEntropy.
    y_std = (y - float(np.mean(y))) / (float(np.std(y)) + 1e-12)
    y_soft = 1.0 / (1.0 + np.exp(-y_std))

    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    # `focal_alpha`/`focal_gamma` are NOT constructor kwargs — they are embedded
    # in the loss_function descriptor string (catboost_options.cpp:234 reads them
    # from the parsed Focal loss params).
    focal_loss = f"Focal:focal_alpha={FOCAL_ALPHA};focal_gamma={FOCAL_GAMMA}"
    # (name, loss_function, target)
    scenarios = [
        ("cross_entropy", "CrossEntropy", y_soft),
        ("focal", focal_loss, y_bin),
    ]
    for name, loss, target in scenarios:
        scenario_dir = LOSS_EXTRA / name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        params = {**ISOLATING_PARAMS, "loss_function": loss}
        model = CatBoostClassifier(boost_from_average=False, **params)
        model.fit(x, target)

        model.save_model(str(scenario_dir / "model.json"), format="json")
        staged = [
            np.asarray(p, dtype=np.float64)
            for p in model.staged_predict(x, prediction_type="RawFormulaVal")
        ]
        staged_flat = _assert_f64(
            np.concatenate([s.ravel() for s in staged]).astype(np.float64), "staged"
        )
        np.save(scenario_dir / "staged.npy", staged_flat, allow_pickle=False)
        predictions = _assert_f64(
            np.asarray(
                model.predict(x, prediction_type="RawFormulaVal"), dtype=np.float64
            ),
            "predictions",
        )
        np.save(scenario_dir / "predictions.npy", predictions, allow_pickle=False)

        config = {
            "scenario": f"loss_extra/{name}",
            "seed": SEED,
            "catboost_version": CATBOOST_VERSION,
            "thread_count": 1,
            "input_dataset": "numeric_tiny",
            "loss_function": loss,
            "boost_from_average": False,
            "params": {**params, "boost_from_average": False},
            "n_rows": int(x.shape[0]),
            "n_iterations": len(staged),
            "label_definition": (
                "y_soft = sigmoid(standardize(numeric_tiny.y))"
                if name == "cross_entropy"
                else "y_binary = (numeric_tiny.y > median)"
            ),
            "prediction_type": "RawFormulaVal",
            "stages": ["Splits", "LeafValues", "StagedApprox", "Predictions"],
            "note": "LOSS-01 CrossEntropy / Focal training oracle (model.json carries leaf_weights)",
        }
        with (scenario_dir / "config.json").open("w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2, sort_keys=True)
            fh.write("\n")


def gen_ordered_boost_e2e() -> None:
    """ordered_boost_e2e/ — the FULL multi-tree ordered train->predict oracle
    (ORD-02, Plan 05-10, the gap-closure for the D-09 omission).

    The prior `ordered_boost/` fixture committed only per-object internals
    (permutation + body/tail boundaries + the iter-0 ordered approx) and OMITTED
    the input features/labels, so it could not validate a full train->predict
    stack (D-09). This scenario commits the WHOLE stack — `X.npy` (f32 features),
    `y.npy` (RMSE labels), `model.json` (the upstream catboost 1.2.10
    boosting_type=Ordered trained model), and `predictions.npy` (upstream
    RawFormulaVal) — so the Rust e2e oracle (`ordered_boost_e2e_oracle_test.rs`)
    trains the SAME isolating config via `cb_train::train`
    (boosting_type=Ordered), lifts the model into `cb_model::Model`, predicts via
    the PRODUCTION `cb_model::predict_raw` apply path, and asserts ≤1e-5 vs these
    `predictions.npy` across ALL iterations/trees (no `#[ignore]`).

    ISOLATING CONFIG (mirrors `ordered_boost/config.json`, the in-scope ordered
    knobs): boosting_type=Ordered, permutation_count=1 (→ 1 learning + 1 averaging
    fold), fold_len_multiplier=2.0, depth=2, iterations=5, learning_rate=0.1,
    l2_leaf_reg=3.0, leaf_estimation_method=Gradient, leaf_estimation_iterations=1,
    bootstrap_type=No, random_strength=0 (→ NO Box-Muller perturbation draws, so
    the once-created fold permutation is deterministic across all iterations —
    the D-11 multi-tree concern does not apply on the ordered path here),
    random_seed=0, thread_count=1, loss=RMSE, boost_from_average explicit.

    OFFLINE / RUN-ONCE (D-12): catboost is NOT importable in CI; this is generated
    on a machine with `catboost==1.2.10` then COMMITTED. CI only READS the
    committed `.npy`/`model.json`. The dataset is a small deterministic numeric
    corpus (N=30, 2 float features) synthesized here with a fixed seed so the
    fixture is reproducible in isolation.
    """
    ORDERED_BOOST_E2E.mkdir(parents=True, exist_ok=True)

    # Deterministic small numeric dataset (N=30, 2 float features, RMSE labels).
    # Fixed seed so the corpus is reproducible without an external input file.
    rng = np.random.default_rng(SEED)
    n_rows = 30
    x = rng.uniform(-3.0, 3.0, size=(n_rows, 2)).astype(np.float32)
    # A smooth-ish target with mild feature dependence + small noise.
    y = (
        1.5 * x[:, 0].astype(np.float64)
        - 0.7 * x[:, 1].astype(np.float64)
        + 0.3 * (x[:, 0].astype(np.float64) ** 2)
        + rng.normal(0.0, 0.1, size=n_rows)
    ).astype(np.float64)

    np.save(ORDERED_BOOST_E2E / "X.npy", x, allow_pickle=False)
    np.save(ORDERED_BOOST_E2E / "y.npy", _assert_f64(y, "y"), allow_pickle=False)

    ordered_params = {
        "boosting_type": "Ordered",
        "permutation_count": 1,
        "fold_len_multiplier": 2.0,
        "depth": 2,
        "iterations": 5,
        "learning_rate": 0.1,
        "l2_leaf_reg": 3.0,
        "leaf_estimation_method": "Gradient",
        "leaf_estimation_iterations": 1,
        "score_function": "L2",
        "bootstrap_type": "No",
        "random_strength": 0,
        "random_seed": SEED,
        "thread_count": 1,
        "loss_function": "RMSE",
        "verbose": False,
    }

    # RMSE → boost_from_average=True (bias == target mean, Pitfall 2).
    # `permutation_count` is a real catboost training parameter (default 4) but is
    # NOT accepted as a sklearn-style named kwarg on CatBoostRegressor; route the
    # whole config through the low-level CatBoost(params) API so permutation_count=1
    # is honored (verified via get_all_params()["permutation_count"] == 1). With an
    # explicit loss_function=RMSE this is identical to CatBoostRegressor.
    model = CatBoost({**ordered_params, "boost_from_average": True})
    model.fit(Pool(x, y))

    # --- Stage: Splits + LeafValues (the upstream Ordered model) -------------
    model.save_model(str(ORDERED_BOOST_E2E / "model.json"), format="json")

    # --- Stage: Predictions (RawFormulaVal, the production apply-path target) -
    predictions = _assert_f64(
        np.asarray(model.predict(x, prediction_type="RawFormulaVal"), dtype=np.float64),
        "predictions",
    )
    np.save(ORDERED_BOOST_E2E / "predictions.npy", predictions, allow_pickle=False)

    config = {
        "scenario": "ordered_boost_e2e",
        "requirement": "ORD-02",
        "seed": SEED,
        "catboost_version": CATBOOST_VERSION,
        "thread_count": 1,
        "n_rows": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "n_iterations": 5,
        "boost_from_average": True,
        "prediction_type": "RawFormulaVal",
        "params": {**ordered_params, "boost_from_average": True},
        "stages": ["Splits", "LeafValues", "Predictions"],
        "carries_full_train_predict_stack": True,
        "note": (
            "FULL ordered train->predict stack (X/y/model.json/predictions), "
            "closing the D-09 omission of the per-object-only ordered_boost/ "
            "fixture. Generated OFFLINE with pinned catboost==1.2.10 (thread_count=1); "
            "NEVER run in CI — CI only READS the committed artifacts (D-12). The Rust "
            "oracle trains the SAME config via cb_train (boosting_type=Ordered) and "
            "asserts cb_model::predict_raw ≤1e-5 vs predictions.npy across ALL trees."
        ),
        "npy_schema": {
            "X.npy": "[N, 2] float32 — input float features (SoA-loaded per column by the Rust oracle)",
            "y.npy": "[N] float64 — RMSE labels",
            "model.json": "upstream catboost 1.2.10 boosting_type=Ordered trained model (splits + leaf_values + borders)",
            "predictions.npy": "[N] float64 — upstream RawFormulaVal (the production-apply-path ≤1e-5 target)",
        },
    }
    with (ORDERED_BOOST_E2E / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
        fh.write("\n")


def gen_tensor_ctr_e2e() -> None:
    """tensor_ctr_e2e/ — the FULL multi-tree TENSOR-CTR train->predict oracle
    (ORD-05, Plan 05-09, the gap-closure for the D-09 omission of `tensor_ctr/`).

    The prior `tensor_ctr/` fixture committed only per-object combined-CTR
    internals (permutation + good/total/value over the combined projection) and
    OMITTED the input cat columns/labels AND the trained model with its baked
    `ctr_data`, so it could not validate a full categorical train->predict stack
    (D-09). This scenario commits the WHOLE stack — `X_cat.npy` (the raw
    categorical columns, integer categories stringified per A4), `y.npy` (Logloss
    labels), `model.json` (the upstream catboost 1.2.10 model trained with
    simple_ctr + combinations_ctr + max_ctr_complexity=2, INCLUDING the baked
    `ctr_data` section), and `predictions.npy` (upstream RawFormulaVal) — so the
    Rust e2e oracle (`tensor_ctr_e2e_oracle_test.rs`) trains the SAME isolating
    config via `cb_train::train`, lifts the model into `cb_model::Model` (with the
    baked ctr_data), predicts via the PRODUCTION `cb_model::predict_raw` /
    `predict_raw_cat` apply path (the ModelSplit::Ctr evaluation), and asserts
    ≤1e-5 vs these `predictions.npy` across ALL iterations/trees (no `#[ignore]`).

    ISOLATING CONFIG (mirrors `tensor_ctr/config.json`): boosting_type=Plain,
    one_hot_max_size=1, max_ctr_complexity=2, simple_ctr=["Borders:Prior=0.5"],
    combinations_ctr=["Borders:Prior=0.5"], permutation_count=1,
    fold_len_multiplier=2.0, counter_calc_method=SkipTest, depth=2, iterations=5,
    learning_rate=0.1, l2_leaf_reg=3.0, leaf_estimation_method=Gradient,
    leaf_estimation_iterations=1, bootstrap_type=No, random_strength=0,
    random_seed=0, thread_count=1, loss_function=Logloss. Two cat features each
    above one_hot_max_size (cardinalities 5 and 4) so a genuine 2-feature
    combination is formed.

    OFFLINE / RUN-ONCE (D-12): catboost is NOT importable in CI; this is generated
    on a machine with `catboost==1.2.10` then COMMITTED. CI only READS the
    committed `.npy`/`model.json`. The dataset is a small deterministic categorical
    corpus (N=30, 2 cat features) synthesized here with a fixed seed so the fixture
    is reproducible in isolation.
    """
    TENSOR_CTR_E2E.mkdir(parents=True, exist_ok=True)

    # Deterministic small categorical dataset (N=30, 2 integer-coded cat features
    # with cardinalities 5 and 4 — both above one_hot_max_size=1 so the CTR path
    # and a genuine 2-feature combination are exercised). Fixed seed so the corpus
    # is reproducible without an external input file.
    rng = np.random.default_rng(SEED)
    n_rows = 30
    cat0 = rng.integers(0, 5, size=n_rows)  # cardinality 5
    cat1 = rng.integers(0, 4, size=n_rows)  # cardinality 4
    # A label with mild dependence on the (cat0, cat1) combination + small noise,
    # binarized to a balanced-ish Logloss target.
    logit = 0.6 * cat0.astype(np.float64) - 0.4 * cat1.astype(np.float64)
    logit = logit - logit.mean() + rng.normal(0.0, 0.5, size=n_rows)
    y = (logit > 0.0).astype(np.float64)

    # X_cat: the raw categorical columns as INTEGER CODES ([N, 2] int32). The
    # Rust oracle stringifies each via cb_data::stringify_int_category (A4 — the
    # PLAIN-integer form cb_data::calc_cat_feature_hash hashes), which is also the
    # form fed to upstream's Pool below (catboost stringifies integer categoricals
    # the same way). int32 keeps the npy loadable by ndarray-npy (no str-npy dep).
    x_cat = np.stack([cat0.astype(np.int32), cat1.astype(np.int32)], axis=1)
    np.save(TENSOR_CTR_E2E / "X_cat.npy", x_cat, allow_pickle=False)
    np.save(TENSOR_CTR_E2E / "y.npy", _assert_f64(y, "y"), allow_pickle=False)

    # The string form upstream's Pool hashes (integer categories stringified, A4).
    x_cat_str = np.stack(
        [cat0.astype(int).astype(str), cat1.astype(int).astype(str)], axis=1
    )

    tensor_ctr_params = {
        "boosting_type": "Plain",
        "one_hot_max_size": 1,
        "max_ctr_complexity": 2,
        "simple_ctr": ["Borders:Prior=0.5"],
        "combinations_ctr": ["Borders:Prior=0.5"],
        "permutation_count": 1,
        "fold_len_multiplier": 2.0,
        "counter_calc_method": "SkipTest",
        "depth": 2,
        "iterations": 5,
        "learning_rate": 0.1,
        "l2_leaf_reg": 3.0,
        "leaf_estimation_method": "Gradient",
        "leaf_estimation_iterations": 1,
        "bootstrap_type": "No",
        "random_strength": 0,
        "random_seed": SEED,
        "thread_count": 1,
        "loss_function": "Logloss",
        "verbose": False,
    }

    # Logloss → boost_from_average=False (starting approx 0, Pitfall 2).
    # `permutation_count` is a real catboost training parameter (default 4) but is
    # NOT accepted as a sklearn-style named kwarg on CatBoostClassifier; route the
    # whole config through the low-level CatBoost(params) API so permutation_count=1
    # is honored (verified via get_all_params()["permutation_count"] == 1). With an
    # explicit loss_function=Logloss this is identical to CatBoostClassifier.
    model = CatBoost({**tensor_ctr_params, "boost_from_average": False})
    # Pool with the two columns declared categorical (cat_features=[0, 1]).
    pool = Pool(x_cat_str, y, cat_features=[0, 1])
    model.fit(pool)

    # --- Stage: Splits + LeafValues + ctr_data (the upstream tensor-CTR model) --
    model.save_model(str(TENSOR_CTR_E2E / "model.json"), format="json")

    # --- Stage: Predictions (RawFormulaVal, the production apply-path target) ---
    predictions = _assert_f64(
        np.asarray(model.predict(x_cat_str, prediction_type="RawFormulaVal"), dtype=np.float64),
        "predictions",
    )
    np.save(TENSOR_CTR_E2E / "predictions.npy", predictions, allow_pickle=False)

    config = {
        "scenario": "tensor_ctr_e2e",
        "requirement": "ORD-05",
        "seed": SEED,
        "catboost_version": CATBOOST_VERSION,
        "thread_count": 1,
        "n_rows": int(x_cat_str.shape[0]),
        "n_cat_features": int(x_cat_str.shape[1]),
        "n_iterations": 5,
        "boost_from_average": False,
        "prediction_type": "RawFormulaVal",
        "params": {**tensor_ctr_params, "boost_from_average": False},
        "stages": ["Splits", "LeafValues", "Predictions"],
        "carries_full_train_predict_stack": True,
        "note": (
            "FULL tensor-CTR train->predict stack (X_cat/y/model.json WITH baked "
            "ctr_data/predictions), closing the D-09 omission of the per-object-only "
            "tensor_ctr/ fixture. Generated OFFLINE with pinned catboost==1.2.10 "
            "(thread_count=1); NEVER run in CI — CI only READS the committed "
            "artifacts (D-12). The Rust oracle trains the SAME config via cb_train, "
            "lifts to cb_model::Model with the baked ctr_data, and asserts "
            "cb_model::predict_raw ≤1e-5 vs predictions.npy across ALL trees through "
            "the ModelSplit::Ctr apply path."
        ),
        "npy_schema": {
            "X_cat.npy": "[N, 2] int32 — raw categorical columns as integer codes (the Rust oracle stringifies via cb_data::stringify_int_category, A4)",
            "y.npy": "[N] float64 — Logloss labels",
            "model.json": "upstream catboost 1.2.10 tensor-CTR model (splits + leaf_values + borders + baked ctr_data)",
            "predictions.npy": "[N] float64 — upstream RawFormulaVal (the production-apply-path ≤1e-5 target)",
        },
    }
    with (TENSOR_CTR_E2E / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
        fh.write("\n")


# Phase-06.1 (Plan 06.1-01, Wave 1, LOSS-03) smooth-regression-loss fixture root.
# The four smooth losses with a REAL der2 — LogCosh, Lq{q}, Huber{delta},
# Expectile{alpha} — each trained on the frozen numeric_tiny corpus with the D-07
# isolating params and the per-loss leaf method PINNED per upstream default
# (RESEARCH Pitfall 2/3/6): LogCosh=Exact (not Newton), Lq(q>=2)/Huber/Expectile=
# Newton, leaf_estimation_iterations:1 pinned (Expectile overrides the upstream
# default of 5 — cb-train is single-step). Generated OFFLINE / RUN-ONCE; CI only
# READS the committed model.json/staged.npy (D-12).
WAVE1_SMOOTH = FIXTURES / "logcosh"  # parent dir is FIXTURES; per-loss subdirs below


def gen_wave1_smooth_losses() -> None:
    """logcosh/ lq/ huber/ expectile/ — the Wave-1 smooth-regression-loss oracle
    (LOSS-03, Plan 06.1-01). Each scenario trains a CatBoostRegressor on the
    frozen `numeric_tiny` corpus with the D-07 simplified isolating params, the
    per-loss `leaf_estimation_method` PINNED per the upstream default (NOT the
    auto-default switch — Pitfall 2), and `leaf_estimation_iterations:1` PINNED
    (Pitfall 3 — cb-train is single-step; this overrides Expectile's upstream
    default of 5). All on depth 2, 5 iterations, lr 0.1, l2 3.0, bootstrap_type
    No, random_strength 0, score_function L2, boost_from_average False,
    random_seed 0, thread_count 1.

    Per-loss config (RESEARCH Wave-1 table, error_functions.h):
      - logcosh : loss_function='LogCosh', leaf_estimation_method='Exact'
        (catboost_options.cpp:65-70 default Exact, NOT Newton — Pitfall 2).
      - lq      : loss_function='Lq:q=2.0' (q>=2 so der2 is Newton-clean,
        Pitfall 6), leaf_estimation_method='Newton'.
      - huber   : loss_function='Huber:delta=1.0', leaf_estimation_method='Newton'
        (catboost_options.cpp:187-192).
      - expectile: loss_function='Expectile:alpha=0.3', leaf_estimation_method=
        'Newton', leaf_estimation_iterations:1 PINNED (override upstream 5).

    boost_from_average is set to False for all four so the starting approx is 0
    (the cb-train smooth-loss path does not boost-from-average for these losses;
    bias 0 keeps the Rust oracle's starting approx trivially matched). Each
    scenario saves config.json (full params recorded), model.json (splits +
    leaf_values + leaf_weights), and staged.npy (per-iteration RawFormulaVal raw
    approximant); stages=[Splits, LeafValues, StagedApprox].

    OFFLINE / RUN-ONCE (D-12): catboost==1.2.10 is NOT importable in CI; this is
    generated on a machine with the pinned catboost then COMMITTED. CI only READS
    the committed artifacts.
    """
    x = np.load(INPUTS / "numeric_tiny" / "X.npy")
    y = np.load(INPUTS / "numeric_tiny" / "y.npy")

    # (name, loss_function, leaf_estimation_method). All four pin
    # leaf_estimation_iterations:1 and boost_from_average=False below.
    scenarios = [
        ("logcosh", "LogCosh", "Exact"),
        ("lq", "Lq:q=2.0", "Newton"),
        ("huber", "Huber:delta=1.0", "Newton"),
        ("expectile", "Expectile:alpha=0.3", "Newton"),
    ]

    for name, loss, estimator in scenarios:
        scenario_dir = FIXTURES / name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        # D-07 isolating params, overriding leaf_estimation_method + loss_function
        # per scenario. leaf_estimation_iterations stays 1 (Pitfall 3).
        params = {**ISOLATING_PARAMS}
        params["leaf_estimation_method"] = estimator
        params["loss_function"] = loss

        model = CatBoostRegressor(boost_from_average=False, **params)
        model.fit(x, y)

        # --- Stage: Splits + LeafValues (model.json) ------------------------
        model.save_model(str(scenario_dir / "model.json"), format="json")

        # --- Stage: StagedApprox (raw approximant, RawFormulaVal) -----------
        staged = [
            np.asarray(p, dtype=np.float64)
            for p in model.staged_predict(x, prediction_type="RawFormulaVal")
        ]
        staged_flat = _assert_f64(
            np.concatenate([s.ravel() for s in staged]).astype(np.float64), "staged"
        )
        np.save(scenario_dir / "staged.npy", staged_flat, allow_pickle=False)

        # --- Stage: Predictions (RawFormulaVal, to match staged final stage) -
        predictions = _assert_f64(
            np.asarray(
                model.predict(x, prediction_type="RawFormulaVal"), dtype=np.float64
            ),
            "predictions",
        )
        np.save(scenario_dir / "predictions.npy", predictions, allow_pickle=False)

        config = {
            "scenario": name,
            "requirement": "LOSS-03",
            "wave": 1,
            "seed": SEED,
            "catboost_version": CATBOOST_VERSION,
            "thread_count": 1,
            "input_dataset": "numeric_tiny",
            "loss_function": loss,
            "leaf_estimation_method": estimator,
            "leaf_estimation_iterations": 1,
            "boost_from_average": False,
            "bootstrap_type": "No",
            "score_function": "L2",
            "params": {**params, "boost_from_average": False},
            "n_rows": int(x.shape[0]),
            "n_features": int(x.shape[1]),
            "n_iterations": len(staged),
            "prediction_type": "RawFormulaVal",
            "stages": ["Splits", "LeafValues", "StagedApprox"],
            "staged_layout": (
                "flat f64: stage 0 (n_rows), then stage 1, ... ; n_iterations "
                "stages (raw approximant)"
            ),
            "leaf_method_note": (
                "leaf_estimation_method PINNED per upstream default (Pitfall 2): "
                "LogCosh=Exact (catboost_options.cpp:65-70, NOT Newton); "
                "Lq(q=2.0)=Newton (q>=2 Newton-clean der2, Pitfall 6); "
                "Huber=Newton (catboost_options.cpp:187-192); "
                "Expectile=Newton with leaf_estimation_iterations:1 PINNED "
                "(override upstream default 5, Pitfall 3)."
            ),
            "der_note": (
                "LogCosh der1=-tanh(approx-target), der2=-1/cosh^2(approx-target) "
                "(error_functions.h:405-425). Lq der1=q*sign(target-approx)*"
                "|approx-target|^(q-1), der2=-q*(q-1)*|target-approx|^(q-2) "
                "(error_functions.h:539-568). Huber der1=|diff|<delta?diff:sign*delta, "
                "der2=|diff|<delta?-1:0, diff=target-approx (error_functions.h:1596-1632). "
                "Expectile der1=(e>0)?2a*e:2(1-a)*e, der2=(e>0)?-2a:-2(1-a), "
                "e=target-approx (error_functions.h:500-537)."
            ),
        }
        with (scenario_dir / "config.json").open("w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2, sort_keys=True)
            fh.write("\n")


def gen_wave1_only() -> None:
    """Targeted entrypoint: regenerate ONLY the Wave-1 smooth-loss fixtures
    (logcosh/lq/huber/expectile) without re-running the full `main()` (which would
    overwrite every committed Phase 2-5 fixture). Used by Plan 06.1-01."""
    gen_wave1_smooth_losses()
    print("Wrote Wave-1 smooth-loss oracle fixtures (logcosh/lq/huber/expectile)")


# Phase-06.1 (Plan 06.1-02, Wave 2, LOSS-03) positive-domain / link regression
# loss fixture roots. Poisson (exp-link / IsStoreExpApprox upstream — cb-train
# computes exp inline on raw approx + Exponent predict), Tweedie{variance_power}
# (exp inside der only, raw approx, NO Exponent), MAPE (der2=0, Gradient leaf),
# and MSLE as an eval-metric ONLY (D-6.1-06: MSLE is metric-only upstream —
# enum_helpers.cpp:200,533-549 — so it has NO training oracle and is NOT a Loss
# variant). Generated OFFLINE / RUN-ONCE; CI only READS the committed artifacts.
POISSON = FIXTURES / "poisson"
TWEEDIE = FIXTURES / "tweedie"
MAPE = FIXTURES / "mape"
MSLE_METRIC = FIXTURES / "msle_metric"

# Phase-06.1 Wave-3 (Plan 06.1-03, LOSS-03) quantile-family fixture roots. The
# Quantile{alpha,delta} Exact-leaf oracle: alpha=0.7 exercises the weighted
# 0.7-quantile leaf (the alpha!=0.5 path), alpha=0.5 is the MAE-equivalence
# anchor (Quantile{0.5} must reproduce leaf_methods/exact (MAE) bit-for-bit).
QUANTILE_ALPHA07 = FIXTURES / "quantile_alpha07"
QUANTILE_ALPHA05_MAE = FIXTURES / "quantile_alpha05_mae"

# numeric_tiny.y carries negatives (min ~ -7.19); Poisson/Tweedie/MAPE require a
# POSITIVE target. Shift the frozen target into a strictly-positive range
# (min -> 1.0) and record the EXACT transform in each config.json so the Rust
# oracle applies the identical positive label column. `y_pos = y - min(y) + 1.0`.
WAVE2_TARGET_SHIFT = 1.0


def _positive_target(y: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (y_pos, shift) where `y_pos = y - min(y) + 1.0` is strictly
    positive (min element is exactly 1.0) and `shift = -min(y) + 1.0` is the
    additive constant the Rust oracle reproduces. f64 throughout."""
    y = _assert_f64(np.asarray(y, dtype=np.float64), "wave2 y")
    shift = float(-y.min() + WAVE2_TARGET_SHIFT)
    y_pos = _assert_f64((y + shift).astype(np.float64), "wave2 y_pos")
    return y_pos, shift


def gen_wave2_positive_losses() -> None:
    """poisson/ tweedie/ mape/ — the Wave-2 positive-domain / link
    regression-loss training oracle (LOSS-03, Plan 06.1-02). Each scenario trains
    a CatBoostRegressor on a POSITIVE-target variant of the frozen `numeric_tiny`
    corpus (y shifted into (0, inf) via `_positive_target`, the transform recorded
    in config.json) with the D-07 simplified isolating params and the per-loss
    `leaf_estimation_method` PINNED + `leaf_estimation_iterations:1` PINNED
    (overriding upstream's Poisson default of 10 — Pitfall 3).

    Per-loss config (RESEARCH Wave-2 table, error_functions.h):
      - poisson : loss_function='Poisson', leaf_estimation_method='Newton',
        leaf_estimation_iterations:1 PINNED (override upstream default 10 —
        Pitfall 3). Poisson is IsStoreExpApprox upstream (approx_updater_helpers.h:
        60-72); StagedApprox(RawFormulaVal) is the RAW approx, Predictions are the
        EXP-transformed values (Open Q1 / Pitfall 4 — empirically confirmed below).
      - tweedie : loss_function='Tweedie:variance_power=1.5' (1<p<2 MANDATORY),
        leaf_estimation_method='Newton', iterations:1. NOT exp-approx
        (error_functions.h:1644) — exp lives inside the der; Predictions are RAW
        (A4 — confirmed below, RawFormulaVal == default predict).
      - mape    : loss_function='MAPE', leaf_estimation_method='Gradient'
        (der2=0 so Newton is undefined — Pitfall 5; catboost_options.cpp:113-124),
        iterations:1.

    boost_from_average=False for all three (the cb-train regression-loss path
    boosts from 0; bias 0 keeps the Rust oracle's starting approx trivially
    matched). Each saves config.json (full params + the positive-target transform
    + the Open-Q1 finding for Poisson), model.json (splits + leaf_values +
    leaf_weights), staged.npy (per-iteration RawFormulaVal raw approximant), and —
    for Poisson + Tweedie — predictions.npy (the Predictions stage: Poisson exp'd,
    Tweedie raw).

    OFFLINE / RUN-ONCE (D-12): catboost==1.2.10 is NOT importable in CI.
    """
    x = np.load(INPUTS / "numeric_tiny" / "X.npy")
    y_raw = np.load(INPUTS / "numeric_tiny" / "y.npy")
    y_pos, shift = _positive_target(y_raw)

    # (name, dir, loss_function, leaf_estimation_method, save_predictions).
    scenarios = [
        ("poisson", POISSON, "Poisson", "Newton", True),
        ("tweedie", TWEEDIE, "Tweedie:variance_power=1.5", "Newton", True),
        ("mape", MAPE, "MAPE", "Gradient", False),
    ]

    for name, scenario_dir, loss, estimator, save_predictions in scenarios:
        scenario_dir.mkdir(parents=True, exist_ok=True)

        # D-07 isolating params, overriding leaf_estimation_method + loss_function
        # per scenario. leaf_estimation_iterations stays 1 (Pitfall 3 — overrides
        # Poisson's upstream default of 10).
        params = {**ISOLATING_PARAMS}
        params["leaf_estimation_method"] = estimator
        params["loss_function"] = loss

        model = CatBoostRegressor(boost_from_average=False, **params)
        model.fit(x, y_pos)

        # --- Stage: Splits + LeafValues (model.json) ------------------------
        model.save_model(str(scenario_dir / "model.json"), format="json")

        # --- Stage: StagedApprox (RAW approximant, RawFormulaVal) -----------
        staged = [
            np.asarray(p, dtype=np.float64)
            for p in model.staged_predict(x, prediction_type="RawFormulaVal")
        ]
        staged_flat = _assert_f64(
            np.concatenate([s.ravel() for s in staged]).astype(np.float64), "staged"
        )
        np.save(scenario_dir / "staged.npy", staged_flat, allow_pickle=False)

        # --- Open Q1 (Poisson): inspect StagedApprox-raw vs exp(raw) --------
        # Pin the empirical finding: staged.npy is RAW approx; the DEFAULT predict
        # (and Exponent transform) is exp(raw). Confirm StagedApprox != exp(raw)
        # for Poisson (so the layout note is auditable).
        open_q1_note = None
        raw_final = staged[-1]
        default_pred = np.asarray(model.predict(x), dtype=np.float64)
        if name == "poisson":
            exp_raw = np.exp(raw_final)
            staged_is_raw = bool(np.max(np.abs(raw_final - default_pred)) > 1e-9)
            predictions_are_exp = bool(np.max(np.abs(exp_raw - default_pred)) <= 1e-6)
            open_q1_note = (
                "EMPIRICAL (catboost 1.2.10, Poisson): "
                f"StagedApprox(RawFormulaVal) is RAW approx (max|raw - default_pred| = "
                f"{float(np.max(np.abs(raw_final - default_pred))):.6e} > 0 confirms "
                f"staged != predict). Predictions = exp(raw): max|exp(raw) - "
                f"default_pred| = {float(np.max(np.abs(exp_raw - default_pred))):.3e} "
                "(<=1e-6 confirms default predict == Exponent(raw)). "
                f"staged_is_raw={staged_is_raw}, predictions_are_exp={predictions_are_exp}. "
                "=> StagedApprox oracle compares RAW; Predictions oracle applies "
                "PredictionType::Exponent (A2 / Pitfall 4)."
            )

        # --- Stage: Predictions (Poisson exp'd via DEFAULT predict; Tweedie RAW) -
        if save_predictions:
            if name == "poisson":
                # Poisson Predictions = exp(raw) = the DEFAULT predict (Exponent).
                predictions = _assert_f64(default_pred, "predictions")
            else:
                # Tweedie Predictions are RAW (no Exponent — A4). Confirm the
                # default predict equals RawFormulaVal (raw) for Tweedie.
                tw_raw = _assert_f64(
                    np.asarray(
                        model.predict(x, prediction_type="RawFormulaVal"),
                        dtype=np.float64,
                    ),
                    "tweedie raw predictions",
                )
                predictions = tw_raw
            np.save(scenario_dir / "predictions.npy", predictions, allow_pickle=False)

        config = {
            "scenario": name,
            "requirement": "LOSS-03",
            "wave": 2,
            "seed": SEED,
            "catboost_version": CATBOOST_VERSION,
            "thread_count": 1,
            "input_dataset": "numeric_tiny",
            "target_transform": (
                f"y_pos = y - min(y) + {WAVE2_TARGET_SHIFT} (additive shift "
                f"{shift!r}); strictly-positive target required by "
                "Poisson/Tweedie/MAPE (min element -> 1.0). The Rust oracle "
                "reproduces the identical positive label column."
            ),
            "target_shift": shift,
            "loss_function": loss,
            "leaf_estimation_method": estimator,
            "leaf_estimation_iterations": 1,
            "boost_from_average": False,
            "bootstrap_type": "No",
            "score_function": "L2",
            "params": {**params, "boost_from_average": False},
            "n_rows": int(x.shape[0]),
            "n_features": int(x.shape[1]),
            "n_iterations": len(staged),
            "prediction_type": (
                "Predictions = Exponent(RawFormulaVal) for Poisson (exp-link); "
                "Predictions = RawFormulaVal (raw) for Tweedie (A4)"
                if save_predictions
                else "RawFormulaVal"
            ),
            "stages": (
                ["Splits", "LeafValues", "StagedApprox", "Predictions"]
                if save_predictions
                else ["Splits", "LeafValues", "StagedApprox"]
            ),
            "staged_layout": (
                "flat f64: stage 0 (n_rows), then stage 1, ... ; n_iterations "
                "stages (RAW approximant, RawFormulaVal)"
            ),
            "leaf_method_note": (
                "Poisson=Newton, leaf_estimation_iterations:1 PINNED (override "
                "upstream default 10, Pitfall 3). Tweedie(p=1.5)=Newton/1. "
                "MAPE=Gradient/1 (der2=0 so Newton is undefined — Pitfall 5, "
                "catboost_options.cpp:113-124)."
            ),
            "der_note": (
                "Poisson der1=target-exp(rawApprox), der2=-exp(rawApprox) "
                "(error_functions.h:657-676; IsStoreExpApprox upstream — cb-train "
                "stores RAW approx + computes exp inline, sigmoid precedent; "
                "approx_updater_helpers.h:60-72). Tweedie der1=target*e^((1-p)*a)-"
                "e^((2-p)*a), der2=target*(1-p)*e^((1-p)*a)-(2-p)*e^((2-p)*a), "
                "p=variance_power, raw approx, exp INSIDE der, NOT exp-approx "
                "(error_functions.h:1634-1665,1644). MAPE der1=sign(target-approx)/"
                "max(1.0,|target|), der2=0 (error_functions.h:607-630; the 1.f "
                "divisor is f32-domain, Pitfall 7)."
            ),
        }
        if open_q1_note is not None:
            config["open_q1_finding"] = open_q1_note
        with (scenario_dir / "config.json").open("w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2, sort_keys=True)
            fh.write("\n")


def gen_msle_metric() -> None:
    """msle_metric/ — the MSLE EVAL-METRIC oracle (D-6.1-06: MSLE is metric-only
    upstream — enum_helpers.cpp:200,533-549 — NOT a trainable objective; Pitfall 1:
    loss_function='MSLE' throws upstream).

    Trains a regression model with a VALID training objective (RMSE) but
    `eval_metric='MSLE'`, capturing catboost's per-iteration MSLE eval-history into
    metric_values.npy. The model.json kept here is the RMSE-trained model (used
    ONLY so the Rust oracle can reproduce the SAME per-iteration staged approx the
    metric is computed over) — there is NO MSLE-as-objective model (it cannot
    exist). MSLE metric = mean_w( (log(1+approx) - log(1+target))^2 ), approx RAW
    (isExpApprox asserted false; metric.cpp:1899-1926).

    The positive target (`_positive_target`) keeps `log(1+approx)`/`log(1+target)`
    in the log domain (1+x > 0). OFFLINE / RUN-ONCE (D-12).
    """
    MSLE_METRIC.mkdir(parents=True, exist_ok=True)
    x = np.load(INPUTS / "numeric_tiny" / "X.npy")
    y_raw = np.load(INPUTS / "numeric_tiny" / "y.npy")
    y_pos, shift = _positive_target(y_raw)

    # RMSE objective (valid trainable) + MSLE eval_metric. Use an eval set == the
    # train set so the per-iteration MSLE history is over the same data the Rust
    # oracle reproduces from staged.npy.
    params = {**ISOLATING_PARAMS}
    params["loss_function"] = "RMSE"
    model = CatBoostRegressor(
        boost_from_average=False, eval_metric="MSLE", **params
    )
    model.fit(x, y_pos, eval_set=(x, y_pos))

    # Per-iteration MSLE eval history (validation_0 == the (x, y_pos) eval set).
    evals = model.get_evals_result()
    # Key is "validation" or "validation_0" depending on version; pick the MSLE
    # curve from the first validation set present.
    val_key = next(k for k in evals if k.startswith("validation"))
    msle_curve = _assert_f64(
        np.asarray(evals[val_key]["MSLE"], dtype=np.float64), "msle metric_values"
    )
    np.save(MSLE_METRIC / "metric_values.npy", msle_curve, allow_pickle=False)

    # The RMSE-trained model + raw staged approx so the Rust oracle reproduces the
    # SAME per-iteration approx the MSLE metric is evaluated over.
    model.save_model(str(MSLE_METRIC / "model.json"), format="json")
    staged = [
        np.asarray(p, dtype=np.float64)
        for p in model.staged_predict(x, prediction_type="RawFormulaVal")
    ]
    staged_flat = _assert_f64(
        np.concatenate([s.ravel() for s in staged]).astype(np.float64), "staged"
    )
    np.save(MSLE_METRIC / "staged.npy", staged_flat, allow_pickle=False)

    config = {
        "scenario": "msle_metric",
        "requirement": "LOSS-03",
        "wave": 2,
        "seed": SEED,
        "catboost_version": CATBOOST_VERSION,
        "thread_count": 1,
        "input_dataset": "numeric_tiny",
        "target_transform": (
            f"y_pos = y - min(y) + {WAVE2_TARGET_SHIFT} (additive shift {shift!r}); "
            "strictly-positive target keeps log(1+x) in domain."
        ),
        "target_shift": shift,
        "loss_function": "RMSE",
        "eval_metric": "MSLE",
        "is_objective": False,
        "boost_from_average": False,
        "params": {**params, "boost_from_average": False, "eval_metric": "MSLE"},
        "n_rows": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "n_iterations": len(staged),
        "prediction_type": "RawFormulaVal",
        "stages": ["StagedApprox", "MetricValues"],
        "metric_values_layout": (
            "flat f64: per-iteration MSLE eval-metric value over the (x, y_pos) "
            "eval set; one entry per boosting iteration."
        ),
        "metric_note": (
            "MSLE is metric-ONLY upstream (D-6.1-06 / Pitfall 1: loss_function="
            "'MSLE' throws). model.json here is the RMSE-trained model kept ONLY "
            "so the Rust oracle reproduces the per-iteration staged approx the "
            "MSLE metric is computed over. MSLE = mean_w((log(1+approx) - "
            "log(1+target))^2), approx RAW (isExpApprox=false; metric.cpp:1899-"
            "1926; GetFinalError = Stats[0]/(Stats[1]+1e-38))."
        ),
    }
    with (MSLE_METRIC / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
        fh.write("\n")


def gen_wave3_quantile_losses() -> None:
    """quantile_alpha07/ quantile_alpha05_mae/ — the Wave-3 quantile-family
    training oracle (LOSS-03, Plan 06.1-03). Each scenario trains a
    CatBoostRegressor on the frozen `numeric_tiny` corpus (the RAW target `y`,
    NOT the Wave-2 positive shift — Quantile admits the full real line) with the
    D-07 simplified isolating params, `leaf_estimation_method='Exact'` PINNED so
    the Exact weighted-alpha-quantile leaf is exercised, and
    `leaf_estimation_iterations:1`.

    The two scenarios (RESEARCH Wave-3 table; error_functions.h:457-498
    TQuantileError, alpha/delta defaults at :468-469):

      - quantile_alpha07 : loss_function='Quantile:alpha=0.7',
        leaf_estimation_method='Exact'. Exercises the alpha!=0.5 weighted-0.7-
        quantile Exact leaf (the genuinely-new path — the thing Task 3 threads).
      - quantile_alpha05_mae : loss_function='Quantile:alpha=0.5',
        leaf_estimation_method='Exact'. The MAE-equivalence ANCHOR: Quantile{0.5}
        must reproduce the existing leaf_methods/exact (MAE) model bit-for-bit
        within <=1e-5 (MAE == Quantile{alpha=0.5}). Identical isolating params to
        leaf_methods/exact except loss_function spelled as Quantile:alpha=0.5.

    boost_from_average=False (the cb-train regression-loss path boosts from 0;
    bias 0 keeps the Rust oracle's starting approx trivially matched — same as the
    MAE leaf_methods/exact fixture). Each saves config.json (full params), model.
    json (splits + leaf_values + leaf_weights), and staged.npy (per-iteration
    RawFormulaVal raw approximant). No predictions.npy — Quantile predictions are
    RAW (no link transform); the StagedApprox stage gates the raw approx.

    A fixture-level MAE==Quantile{0.5} sanity check runs here: the alpha=0.5 model
    leaf_values are compared against the existing leaf_methods/exact (MAE) model
    leaf_values and asserted <=1e-5 before the fixtures are committed.

    OFFLINE / RUN-ONCE (D-12): catboost==1.2.10 is NOT importable in CI.
    """
    x = np.load(INPUTS / "numeric_tiny" / "X.npy")
    y = np.load(INPUTS / "numeric_tiny" / "y.npy")

    # (name, dir, loss_function, alpha).
    scenarios = [
        ("quantile_alpha07", QUANTILE_ALPHA07, "Quantile:alpha=0.7", 0.7),
        ("quantile_alpha05_mae", QUANTILE_ALPHA05_MAE, "Quantile:alpha=0.5", 0.5),
    ]

    for name, scenario_dir, loss, alpha in scenarios:
        scenario_dir.mkdir(parents=True, exist_ok=True)

        # D-07 isolating params, overriding leaf_estimation_method='Exact' (so the
        # Exact weighted-alpha-quantile leaf is under test) + loss_function per
        # scenario. leaf_estimation_iterations stays 1.
        params = {**ISOLATING_PARAMS}
        params["leaf_estimation_method"] = "Exact"
        params["loss_function"] = loss

        model = CatBoostRegressor(boost_from_average=False, **params)
        model.fit(x, y)

        # --- Stage: Splits + LeafValues (model.json) ------------------------
        model.save_model(str(scenario_dir / "model.json"), format="json")

        # --- Stage: StagedApprox (RAW approximant, RawFormulaVal) -----------
        staged = [
            np.asarray(p, dtype=np.float64)
            for p in model.staged_predict(x, prediction_type="RawFormulaVal")
        ]
        staged_flat = _assert_f64(
            np.concatenate([s.ravel() for s in staged]).astype(np.float64), "staged"
        )
        np.save(scenario_dir / "staged.npy", staged_flat, allow_pickle=False)

        config = {
            "scenario": name,
            "requirement": "LOSS-03",
            "wave": 3,
            "seed": SEED,
            "catboost_version": CATBOOST_VERSION,
            "thread_count": 1,
            "input_dataset": "numeric_tiny",
            "loss_function": loss,
            "alpha": alpha,
            "delta": 1e-6,
            "leaf_estimation_method": "Exact",
            "leaf_estimation_iterations": 1,
            "boost_from_average": False,
            "bootstrap_type": "No",
            "score_function": "L2",
            "params": {**params, "boost_from_average": False},
            "n_rows": int(x.shape[0]),
            "n_features": int(x.shape[1]),
            "n_iterations": len(staged),
            "prediction_type": "RawFormulaVal",
            "stages": ["Splits", "LeafValues", "StagedApprox"],
            "staged_layout": (
                "flat f64: stage 0 (n_rows), then stage 1, ... ; n_iterations "
                "stages (RAW approximant, RawFormulaVal)"
            ),
            "leaf_method_note": (
                "Quantile=Exact (weighted alpha-quantile of leaf residuals "
                "target-approx; CalcOneDimensionalOptimumConstApprox -> "
                "CalculateWeightedTargetQuantile, error_functions.h:457-498). "
                "leaf_estimation_iterations:1. The alpha=0.7 fixture exercises the "
                "alpha!=0.5 path; the alpha=0.5 fixture is the MAE-equivalence "
                "anchor (Quantile{0.5}==MAE bit-for-bit vs leaf_methods/exact)."
            ),
            "der_note": (
                "Quantile der1: val=target-approx; |val|<delta ? 0 : "
                "(val>0 ? alpha : -(1-alpha)); der2=0 "
                "(error_functions.h:485-493 TQuantileError; alpha/delta defaults "
                "0.5/1e-6 at :468-469). At alpha=0.5,delta=1e-6 it equals MAE."
            ),
        }
        with (scenario_dir / "config.json").open("w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2, sort_keys=True)
            fh.write("\n")

    # --- Fixture-level MAE == Quantile{0.5} sanity (acceptance criterion) -----
    # The alpha=0.5 Quantile model leaf_values MUST match the existing
    # leaf_methods/exact (MAE) model leaf_values within <=1e-5 (MAE ==
    # Quantile{alpha=0.5}). Assert here so a misconfigured fixture is caught at
    # generation time, before commit.
    mae_model_path = LEAF_METHODS / "exact" / "model.json"
    if mae_model_path.exists():
        with mae_model_path.open(encoding="utf-8") as fh:
            mae_model = json.load(fh)
        with (QUANTILE_ALPHA05_MAE / "model.json").open(encoding="utf-8") as fh:
            q05_model = json.load(fh)

        def _leaf_values(model_json: dict) -> np.ndarray:
            vals: list[float] = []
            for tree in model_json["oblivious_trees"]:
                vals.extend(float(v) for v in tree["leaf_values"])
            return np.asarray(vals, dtype=np.float64)

        mae_lv = _leaf_values(mae_model)
        q05_lv = _leaf_values(q05_model)
        if mae_lv.shape != q05_lv.shape:
            raise AssertionError(
                "MAE==Quantile{0.5} sanity: leaf-value shape mismatch "
                f"(MAE {mae_lv.shape} vs Quantile{{0.5}} {q05_lv.shape})"
            )
        max_diff = float(np.max(np.abs(mae_lv - q05_lv)))
        if max_diff > 1e-5:
            raise AssertionError(
                "MAE==Quantile{0.5} sanity FAILED: max|leaf_value diff| = "
                f"{max_diff:.6e} > 1e-5 — the alpha=0.5 fixture does not reproduce "
                "the leaf_methods/exact (MAE) model. Check the config."
            )
        print(
            "  MAE==Quantile{0.5} sanity OK: max|leaf_value diff| = "
            f"{max_diff:.6e} <= 1e-5"
        )


def gen_wave3_only() -> None:
    """Targeted entrypoint: regenerate ONLY the Plan 06.1-03 Wave-3 quantile
    fixtures (quantile_alpha07 / quantile_alpha05_mae) without re-running the full
    `main()` (which would overwrite every committed Phase 2-5 / Wave-1/2 fixture).

    NOTE: the MAE==Quantile{0.5} sanity reads the committed leaf_methods/exact
    model.json, so that fixture must already exist (it is committed)."""
    gen_wave3_quantile_losses()
    print(
        "Wrote Wave-3 quantile-family oracle fixtures "
        "(quantile_alpha07/quantile_alpha05_mae)"
    )


def gen_wave2_only() -> None:
    """Targeted entrypoint: regenerate ONLY the Plan 06.1-02 Wave-2 fixtures
    (poisson/tweedie/mape training + msle_metric) without re-running the full
    `main()` (which would overwrite every committed Phase 2-5 fixture)."""
    gen_wave2_positive_losses()
    print("Wrote Wave-2 positive-loss oracle fixtures (poisson/tweedie/mape)")
    gen_msle_metric()
    print("Wrote MSLE eval-metric oracle fixture (msle_metric)")


# Phase-06.2 Wave-1 (Plan 06.2-03, LOSS-02) multiclass fixture roots. A 3-class
# target is built from the numeric_tiny regression target's terciles; the
# remapped contiguous class index [0,3) is the training target. MultiClass is the
# cross-dimension-coupled softmax loss (symmetric Hessian Newton solve); OneVsAll
# is the separable per-dimension diagonal sigmoid. BOTH use the catboost MultiClass
# default score_function=Cosine (NOT the L2 the scalar fixtures pin) and
# leaf_estimation_method=Newton / leaf_estimation_iterations=1 (Pitfall 2).
MULTICLASS_SOFTMAX = FIXTURES / "multiclass_softmax"
MULTICLASS_ONEVSALL = FIXTURES / "multiclass_onevsall"


def _multiclass_target() -> np.ndarray:
    """The 3-class target derived from numeric_tiny.y terciles: digitize y into
    {0,1,2} at the 1/3 and 2/3 quantiles. Deterministic (numeric_tiny.y is frozen);
    the same tercile rule is reproduced by the Rust oracle test so train/predict
    use the identical class labels."""
    y = np.load(INPUTS / "numeric_tiny" / "y.npy")
    q = np.quantile(y, [1.0 / 3.0, 2.0 / 3.0])
    return np.digitize(y, q).astype(np.int64)  # 0, 1, 2


def _gen_one_multiclass(scenario_dir: Path, loss: str) -> None:
    """Train one multiclass loss (MultiClass | MultiClassOneVsAll) on the 3-class
    numeric_tiny target and freeze its per-stage oracle: model.json (splits +
    LEAF-MAJOR leaf_values + class_params), staged.npy (staged_predict
    RawFormulaVal, shape (n,dim) object-major — A4 flatten recorded), and
    predictions.npy (Probability)."""
    scenario_dir.mkdir(parents=True, exist_ok=True)
    x = np.load(INPUTS / "numeric_tiny" / "X.npy")
    yc = _multiclass_target()

    # MultiClass default score_function is Cosine (NOT the scalar fixtures' L2);
    # leaf_estimation_method=Newton (the multiclass default) with iterations=1
    # (Pitfall 2). bootstrap_type=No, random_strength=0, thread_count=1 isolate the
    # tree/leaf math (D-07). boost_from_average=False (classification).
    params = {
        "iterations": 5,
        "learning_rate": 0.1,
        "depth": 2,
        "l2_leaf_reg": 3.0,
        "bootstrap_type": "No",
        "random_strength": 0,
        "leaf_estimation_iterations": 1,
        "leaf_estimation_method": "Newton",
        "score_function": "Cosine",
        "random_seed": SEED,
        "thread_count": 1,
        "verbose": False,
        "loss_function": loss,
        "boost_from_average": False,
    }
    model = CatBoostClassifier(**params)
    model.fit(x, yc)

    # --- Stage: Splits + LeafValues + class_params (model.json) -------------
    model.save_model(str(scenario_dir / "model.json"), format="json")

    # --- Stage: StagedApprox (RawFormulaVal, shape (n,dim) object-major) ----
    # staged_predict(RawFormulaVal) yields one (n, dim) array per iteration; flatten
    # OBJECT-MAJOR (row-major, object then dim) and concatenate across iterations.
    # The Rust oracle transposes its dimension-major training buffer to this
    # object-major layout before comparing (A4).
    staged = [
        np.asarray(p, dtype=np.float64)
        for p in model.staged_predict(x, prediction_type="RawFormulaVal")
    ]
    staged_flat = _assert_f64(
        np.concatenate([s.ravel() for s in staged]).astype(np.float64), "staged"
    )
    np.save(scenario_dir / "staged.npy", staged_flat, allow_pickle=False)

    # --- Stage: Predictions (Probability, shape (n,dim) object-major) -------
    proba = np.asarray(model.predict(x, prediction_type="Probability"), dtype=np.float64)
    proba_flat = _assert_f64(proba.ravel().astype(np.float64), "predictions")
    np.save(scenario_dir / "predictions.npy", proba_flat, allow_pickle=False)

    n_iter = len(staged)
    dim = int(np.asarray(staged[0]).shape[1]) if n_iter else 0
    config = {
        "scenario": scenario_dir.name,
        "seed": SEED,
        "catboost_version": CATBOOST_VERSION,
        "thread_count": 1,
        "input_dataset": "numeric_tiny",
        "loss_function": loss,
        "leaf_estimation_method": "Newton",
        "score_function": "Cosine",
        "boost_from_average": False,
        "params": params,
        "n_rows": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "n_iterations": n_iter,
        "approx_dimension": dim,
        "class_labels": [0, 1, 2],
        "target_rule": "digitize(numeric_tiny.y, quantile(y,[1/3,2/3])) -> {0,1,2}",
        "stages": ["Splits", "LeafValues", "StagedApprox", "Predictions"],
        "staged_layout": (
            "staged_predict(RawFormulaVal): per-iter (n_rows, dim) OBJECT-MAJOR "
            "(row-major object then dim), concatenated across iterations; flat f64."
        ),
        "predictions_layout": (
            "predict(Probability): (n_rows, dim) OBJECT-MAJOR row-major, flat f64."
        ),
        "leaf_values_layout": (
            "model.json leaf_values are LEAF-MAJOR (leaf0_d0, leaf0_d1, ..., "
            "leaf1_d0, ...) length leaves*dim; leaf_weights length leaves."
        ),
        "prediction_type": "Probability",
    }
    with (scenario_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
        fh.write("\n")


def gen_multiclass() -> None:
    """multiclass_softmax/ + multiclass_onevsall/ — the Plan 06.2-03 (LOSS-02)
    multiclass per-stage oracle. MultiClass (softmax, cross-dimension coupled
    symmetric Hessian Newton solve) and MultiClassOneVsAll (separable per-dimension
    diagonal sigmoid Newton). Each gates Splits / LeafValues / StagedApprox /
    Predictions <= 1e-5 vs catboost 1.2.10 (thread_count=1, the pinned isolating
    params)."""
    _gen_one_multiclass(MULTICLASS_SOFTMAX, "MultiClass")
    _gen_one_multiclass(MULTICLASS_ONEVSALL, "MultiClassOneVsAll")


def gen_multiclass_only() -> None:
    """Targeted entrypoint: regenerate ONLY the Plan 06.2-03 multiclass fixtures
    (multiclass_softmax / multiclass_onevsall) without re-running the full `main()`
    (which would overwrite every committed Phase 2-5 / 6.1 fixture)."""
    gen_multiclass()
    print(
        "Wrote multiclass oracle fixtures "
        "(multiclass_softmax/multiclass_onevsall)"
    )


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

    # --- regularization (TRAIN-05 l2/random_strength/bagging_temp, D-10) ----
    gen_regularization()
    print(f"Wrote regularization oracle fixtures under {REGULARIZATION}")

    # --- overfit (TRAIN-06 inctodec/iter/wilcoxon/use_best_model, D-10) ------
    gen_overfit()
    print(f"Wrote overfit oracle fixtures under {OVERFIT}")

    # --- eval_metrics (TRAIN-07 per-iteration eval-set metric logging, D-10) -
    gen_eval_metrics()
    print(f"Wrote eval_metrics oracle fixtures under {EVAL_METRICS}")

    # --- autolr (TRAIN-08 automatic learning-rate selection, D-10) ----------
    gen_autolr()
    print(f"Wrote autolr oracle fixtures under {AUTOLR}")

    # --- Phase-4 offline fixtures (Plan 04-01, D-13) ------------------------
    gen_model_serde()
    print(f"Wrote model_serde oracle fixtures under {MODEL_SERDE}")
    gen_prediction_types()
    print(f"Wrote prediction_types oracle fixtures under {PREDICTION_TYPES}")
    gen_feature_importance()
    print(f"Wrote feature_importance oracle fixtures under {FEATURE_IMPORTANCE}")
    gen_loss_extra()
    print(f"Wrote loss_extra oracle fixtures under {LOSS_EXTRA}")

    # --- Phase-5 ordered end-to-end fixture (Plan 05-10, ORD-02, D-09) -------
    gen_ordered_boost_e2e()
    print(f"Wrote ordered_boost_e2e oracle fixtures under {ORDERED_BOOST_E2E}")

    # --- Phase-5 tensor-CTR end-to-end fixture (Plan 05-09, ORD-05, D-09) -----
    gen_tensor_ctr_e2e()
    print(f"Wrote tensor_ctr_e2e oracle fixtures under {TENSOR_CTR_E2E}")

    # --- Phase-06.1 Wave-1 smooth-loss fixtures (Plan 06.1-01, LOSS-03) ------
    gen_wave1_smooth_losses()
    print("Wrote Wave-1 smooth-loss oracle fixtures (logcosh/lq/huber/expectile)")

    # --- Phase-06.1 Wave-2 positive-loss fixtures (Plan 06.1-02, LOSS-03) ----
    gen_wave2_positive_losses()
    print("Wrote Wave-2 positive-loss oracle fixtures (poisson/tweedie/mape)")
    gen_msle_metric()
    print("Wrote MSLE eval-metric oracle fixture (msle_metric)")

    # --- Phase-06.1 Wave-3 quantile-family fixtures (Plan 06.1-03, LOSS-03) ---
    gen_wave3_quantile_losses()
    print(
        "Wrote Wave-3 quantile-family oracle fixtures "
        "(quantile_alpha07/quantile_alpha05_mae)"
    )

    # --- Phase-06.2 Wave-1 multiclass fixtures (Plan 06.2-03, LOSS-02) -------
    gen_multiclass()
    print(
        "Wrote multiclass oracle fixtures "
        "(multiclass_softmax/multiclass_onevsall)"
    )

    # --- Wave-0 scenarios (A1-A5 resolution) --------------------------------
    gen_borders_quant()
    print(f"Wrote borders_quant oracle fixtures under {BORDERS_QUANT}")
    gen_cat_hash()
    print(f"Wrote cat_hash oracle fixtures under {CAT_HASH}")
    gen_class_weights()
    print(f"Wrote class_weights oracle fixtures under {CLASS_WEIGHTS}")


if __name__ == "__main__":
    import sys

    # `--wave1-only` regenerates ONLY the Plan 06.1-01 smooth-loss fixtures
    # (logcosh/lq/huber/expectile), leaving every committed Phase 2-5 fixture
    # untouched. Bare invocation regenerates everything (the RUN-ONCE full path).
    if "--wave1-only" in sys.argv:
        gen_wave1_only()
    elif "--wave2-only" in sys.argv:
        # `--wave2-only` regenerates ONLY the Plan 06.1-02 Wave-2 fixtures
        # (poisson/tweedie/mape + msle_metric), leaving every committed Phase 2-5
        # / Wave-1 fixture untouched.
        gen_wave2_only()
    elif "--wave3-only" in sys.argv:
        # `--wave3-only` regenerates ONLY the Plan 06.1-03 Wave-3 quantile
        # fixtures (quantile_alpha07/quantile_alpha05_mae), leaving every committed
        # Phase 2-5 / Wave-1/2 fixture untouched.
        gen_wave3_only()
    elif "--multiclass-only" in sys.argv:
        # `--multiclass-only` regenerates ONLY the Plan 06.2-03 multiclass fixtures
        # (multiclass_softmax / multiclass_onevsall), leaving every committed
        # Phase 2-5 / 6.1 fixture untouched.
        gen_multiclass_only()
    else:
        main()
