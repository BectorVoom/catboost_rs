"""Freeze the standalone quantization borders for EVERY `feature_border_type`.

Parity target for `cb_data::select_borders` (the border-selection family behind
the `feature_border_type` parameter). Upstream exposes seven binarizers
(`EBorderSelectionType`, authoritative list probed from the installed wheel):

    Median, GreedyLogSum, UniformAndQuantiles, MinEntropy,
    MaxLogSum, Uniform, GreedyMinEntropy

Only `GreedyLogSum` was implemented before this wave, and the existing
`borders_quant/` fixture covers it at the default `border_count=254`.

# Why a new fixture instead of extending borders_quant

`borders_quant` runs at `border_count=254` over 50-row columns with 50 unique
values. The budget therefore EXCEEDS the number of representable splits, so
every binarizer saturates to (almost) the same ~49-border answer and the fixture
cannot tell the seven algorithms apart. A parity fixture that passes for the
wrong implementation is worse than none, so this generator deliberately runs
UNDER-BUDGET (`border_count` well below the unique-value count), which is the
regime where the penalty function and the search strategy actually decide the
output.

# Ground truth

`Pool.quantize(border_count, feature_border_type, nan_mode)` followed by
`save_quantization_borders(path)` — the STANDALONE binarizer, exactly as
`gen_fixtures._quantization_borders` does it, NOT a trained model's
`get_borders()` (which returns a training-pruned subset; that was the Wave-0 bug
recorded in `borders_quant/config.json`).

Run:  python3 crates/cb-oracle/generator/gen_border_type_fixtures.py
"""

import json
import os
import tempfile

import numpy as np
from catboost import Pool

CATBOOST_VERSION = "1.2.10"

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "fixtures")
OUT_DIR = os.path.join(FIXTURES, "border_types")
INPUTS = os.path.join(FIXTURES, "inputs")

# The authoritative legal set, probed from catboost 1.2.10 by passing a bogus
# value and reading the enum parser's rejection message:
#   "Key 'ZzBogusValue' not found in enum EBorderSelectionType. Valid options
#    are: 'Median', 'GreedyLogSum', 'UniformAndQuantiles', 'MinEntropy',
#    'MaxLogSum', 'Uniform', 'GreedyMinEntropy'."
BORDER_TYPES = [
    "Median",
    "GreedyLogSum",
    "UniformAndQuantiles",
    "MinEntropy",
    "MaxLogSum",
    "Uniform",
    "GreedyMinEntropy",
]

# The dense input this generator synthesizes (a plain numeric_tiny-shaped file
# under inputs/, so the Rust side loads it through the same X.npy convention).
DENSE_DATASET = "borders_dense"
DENSE_SEED = 20260814
DENSE_ROWS = 2000

# The DISCRIMINATING corpus. See `runs_column` for why it exists.
RUNS_DATASET = "borders_runs"
RUNS_ROWS = 1200
# Column seeds chosen by an exhaustive search over 60 candidates for maximum
# pairwise distinctness of the seven binarizers (see the module doc).
RUNS_SEEDS = [51, 33, 15, 28]


def synthesize_dense_input() -> np.ndarray:
    """A 2000x4 f64 matrix whose columns span distributions that DISCRIMINATE
    the binarizers: uniform, heavy-tailed lognormal, a discrete low-cardinality
    column, and a bimodal mixture. Equal-frequency (Median) and equal-width
    (Uniform) answers diverge sharply on the skewed columns, and the
    low-cardinality column exercises the duplicate-value grouping that the exact
    (MinEntropy / MaxLogSum) binarizers do before their DP.
    """
    rng = np.random.default_rng(DENSE_SEED)
    uniform = rng.uniform(-5.0, 5.0, DENSE_ROWS)
    lognormal = rng.lognormal(mean=0.0, sigma=1.5, size=DENSE_ROWS)
    discrete = rng.integers(0, 12, DENSE_ROWS).astype(np.float64)
    bimodal = np.concatenate(
        [
            rng.normal(-3.0, 0.4, DENSE_ROWS // 2),
            rng.normal(4.0, 1.2, DENSE_ROWS - DENSE_ROWS // 2),
        ]
    )
    rng.shuffle(bimodal)
    x = np.column_stack([uniform, lognormal, discrete, bimodal])
    return np.ascontiguousarray(x, dtype=np.float64)


def runs_column(seed: int, rows: int = RUNS_ROWS) -> np.ndarray:
    """One column of a small unique-value set with WILDLY UNEVEN repeat counts.

    This shape is what separates the two GREEDY penalties (MaxSumLog vs
    MinEntropy) and the two EXACT ones. On evenly-spread data they provably
    coincide: with unit weights and a bin of `n` objects split into `l + r = n`,

        MaxSumLog  score = log(l+eps) + log(r+eps) - log(n+eps)
        MinEntropy score = n*log(n) - l*log(l) - r*log(r)

    are BOTH maximized at the balanced split `l = r`, and both increase
    monotonically with bin size, so the greedy heap pops the same bins and picks
    the same split points. They can only diverge when the achievable split
    positions are asymmetric -- i.e. heavy, unevenly-placed duplicate runs --
    because only then do the two scores trade "bin size" against "split
    imbalance" at different rates.

    Without this corpus the fixture would be VACUOUS for GreedyMinEntropy and
    MinEntropy: on every evenly-spread dataset tried they are byte-identical to
    GreedyLogSum / MaxLogSum respectively, so the test would pass even if those
    two binarizers were implemented with the wrong penalty.
    """
    rng = np.random.default_rng(seed)
    n_uniq = int(rng.integers(10, 45))
    uniq = np.sort(rng.uniform(0.0, 100.0, n_uniq))
    weights = rng.exponential(1.0, n_uniq)
    counts = np.maximum(1, np.round(weights / weights.sum() * rows).astype(int))
    while counts.sum() > rows:
        counts[np.argmax(counts)] -= 1
    while counts.sum() < rows:
        counts[np.argmax(counts)] += 1
    return np.repeat(uniq, counts).astype(np.float64)


def synthesize_runs_input() -> np.ndarray:
    cols = [runs_column(s) for s in RUNS_SEEDS]
    return np.ascontiguousarray(np.column_stack(cols), dtype=np.float64)


# Corpora that live outside inputs/ (their own fixture directory owns them).
EXTERNAL_INPUTS = {
    # The SPEC-OH-31 byte-identity corpus. It quantizes 512 unique values into 32
    # borders, so its border budget BINDS hard -- which is exactly why the greedy
    # tie-break bug moved its frozen `.cbm`. Freezing catboost's borders for it
    # turns "the baseline changed" into "the baseline changed TOWARD upstream".
    "float_only_byte_identity": os.path.join(
        FIXTURES, "float_only_byte_identity", "inputs", "X.npy"
    ),
}


def load_input(dataset: str) -> np.ndarray:
    path = EXTERNAL_INPUTS.get(dataset)
    if path is None:
        path = os.path.join(INPUTS, dataset, "X.npy")
    return np.load(path)


def standalone_borders(x, border_count, border_type, nan_mode):
    """The RAW standalone binarizer output, per feature index."""
    # Pool needs a label column; the binarizer never reads it (border selection
    # is unsupervised), so a zero vector keeps the fixture target-independent.
    y = np.zeros(x.shape[0], dtype=np.float64)
    pool = Pool(x, y)
    pool.quantize(
        border_count=border_count,
        feature_border_type=border_type,
        nan_mode=nan_mode,
    )
    path = tempfile.mktemp(suffix=".borders.tsv")
    pool.save_quantization_borders(path)
    borders = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            # "<feature>\t<border>" or "<feature>\t<border>\t<nan_mode>".
            parts = line.split("\t")
            borders.setdefault(int(parts[0]), []).append(float(parts[1]))
    os.unlink(path)
    return borders


def flatten(borders, n_features):
    """Flat f64 border vector + per-feature counts, the borders_quant layout."""
    flat = []
    counts = []
    for fi in range(n_features):
        col = sorted(borders.get(fi, []))
        flat.extend(col)
        counts.append(len(col))
    return np.asarray(flat, dtype=np.float64), np.asarray(counts, dtype=np.float64)


# (dataset, border_count, nan_mode) — every cell is run for all 7 types.
SCENARIOS = [
    # Under-budget on a 50-row / 50-unique corpus: 8 and 32 borders both sit
    # well below the 49 representable splits, so the algorithms disagree.
    ("numeric_tiny", 8, "Min"),
    ("numeric_tiny", 32, "Min"),
    # The NaN corpus at a small budget: covers sentinel x border-type interaction.
    ("numeric_nan", 8, "Min"),
    # The dense corpus: 2000 rows, skewed + low-cardinality columns. 16 exercises
    # the search hard; 128 approaches (but stays under) the discrete column's
    # saturation point.
    (DENSE_DATASET, 16, "Min"),
    (DENSE_DATASET, 128, "Min"),
    # The DISCRIMINATING corpus (uneven duplicate runs). These are the only cells
    # where GreedyMinEntropy differs from GreedyLogSum and MinEntropy from
    # MaxLogSum, so they are what makes the fixture non-vacuous for the two
    # MinEntropy-penalty binarizers.
    (RUNS_DATASET, 8, "Min"),
    (RUNS_DATASET, 16, "Min"),
    (RUNS_DATASET, 32, "Min"),
    # The SPEC-OH-31 byte-identity corpus at the exact budget its fixture uses
    # (512 unique values -> 32 borders). See EXTERNAL_INPUTS.
    ("float_only_byte_identity", 32, "Min"),
]


def write_input(dataset: str, x: np.ndarray, meta: dict) -> None:
    """Materialize a synthesized corpus under inputs/ so the Rust test loads it
    the same way as every other corpus (X.npy / y.npy / config.json)."""
    directory = os.path.join(INPUTS, dataset)
    os.makedirs(directory, exist_ok=True)
    np.save(os.path.join(directory, "X.npy"), x)
    np.save(os.path.join(directory, "y.npy"), np.zeros(x.shape[0], dtype=np.float64))
    meta = dict(meta)
    meta["dataset"] = dataset
    meta["y"] = "all zeros - border selection is unsupervised and never reads the label"
    with open(os.path.join(directory, "config.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    dense_x = synthesize_dense_input()
    write_input(
        DENSE_DATASET,
        dense_x,
        {
            "seed": DENSE_SEED,
            "rows": DENSE_ROWS,
            "columns": [
                "uniform(-5,5)",
                "lognormal(0,1.5)",
                "discrete integers 0..11",
                "bimodal normal mixture(-3,0.4)+(4,1.2)",
            ],
            "purpose": (
                "Spread corpus for feature_border_type: skewed and "
                "low-cardinality columns make equal-frequency, equal-width "
                "and penalty-optimal binarizers disagree."
            ),
        },
    )

    runs_x = synthesize_runs_input()
    write_input(
        RUNS_DATASET,
        runs_x,
        {
            "seeds": RUNS_SEEDS,
            "rows": RUNS_ROWS,
            "columns": [
                "small unique-value set with wildly uneven duplicate run lengths"
            ]
            * len(RUNS_SEEDS),
            "purpose": (
                "The DISCRIMINATING corpus. On evenly-spread data the MaxSumLog "
                "and MinEntropy penalties provably coincide (both peak at the "
                "balanced split and both grow with bin size), so GreedyMinEntropy "
                "is byte-identical to GreedyLogSum and MinEntropy to MaxLogSum. "
                "Uneven duplicate runs break that tie, which is what makes the "
                "border_types fixture non-vacuous for those two binarizers. See "
                "generator/gen_border_type_fixtures.py::runs_column."
            ),
        },
    )

    synthesized = {DENSE_DATASET: dense_x, RUNS_DATASET: runs_x}

    scenarios_meta = {}
    for dataset, border_count, nan_mode in SCENARIOS:
        x = synthesized.get(dataset)
        if x is None:
            x = load_input(dataset)
        n_features = x.shape[1]
        for border_type in BORDER_TYPES:
            borders = standalone_borders(x, border_count, border_type, nan_mode)
            flat, counts = flatten(borders, n_features)
            stem = "%s.bc%d.%s" % (dataset, border_count, border_type)
            np.save(os.path.join(OUT_DIR, stem + ".borders.npy"), flat)
            np.save(os.path.join(OUT_DIR, stem + ".borders_per_feature.npy"), counts)
            scenarios_meta[stem] = {
                "dataset": dataset,
                "border_count": border_count,
                "feature_border_type": border_type,
                "nan_mode": nan_mode,
                "n_features": int(n_features),
                "n_borders_per_feature": [int(c) for c in counts],
                "n_borders_total": int(counts.sum()),
            }
            print(
                "%-46s %s"
                % (stem, [int(c) for c in counts])
            )

    with open(os.path.join(OUT_DIR, "config.json"), "w") as fh:
        json.dump(
            {
                "scenario": "border_types",
                "catboost_version": CATBOOST_VERSION,
                "borders_source": "standalone Pool.quantize(...).save_quantization_borders()",
                "borders_layout": (
                    "<stem>.borders.npy = flat f64 (feature 0 borders, then feature 1, ...); "
                    "<stem>.borders_per_feature.npy = per-feature counts. "
                    "stem = <dataset>.bc<border_count>.<feature_border_type>"
                ),
                "border_types": BORDER_TYPES,
                "under_budget_rationale": (
                    "Every cell runs with border_count BELOW the column's unique-value "
                    "count. At or above saturation all seven binarizers collapse to the "
                    "same answer and the fixture would pass for a wrong implementation."
                ),
                "scenarios": scenarios_meta,
            },
            fh,
            indent=2,
            sort_keys=True,
        )
    print("\nwrote %d border sets to %s" % (len(scenarios_meta), OUT_DIR))


if __name__ == "__main__":
    main()
