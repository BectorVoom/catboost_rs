"""Generate the BinarizedTargetMeanValue-CTR device-parity fixture (DCTR-14 / T07).

FROZEN GENERATOR — the committed artifacts under this directory are the GROUND TRUTH the
device BTMV CTR path (Track B: the f32 `TCtrMeanHistory::Sum` accumulator plus the
unchanged `binarize_ctr_column_resident` quantizer) is compared against at <=1e-5. CI does
NOT run this script and never regenerates these fixtures.

# Why this fixture exists (DCTR-14)

`ctr_btmv_simple/` already isolates simple BTMV end-to-end, but it is a categorical-ONLY
pool (zero float columns). A cat-only pool gives `device_n_float == 0`, so
`has_any_scorable_feature` declines and the fit can NEVER commit to the device
(`boosting.rs:3284-3286`). This is the first BTMV fixture that is device-REACHABLE: two
float columns quantized at `border_count=15` alongside one cardinality-6 cat column routed
to CTR by `one_hot_max_size=1`.

# Deltas from `ctr_device_mixed/` (the proven device-reachable CTR recipe)

  - `simple_ctr=["BinarizedTargetMeanValue:Prior=0.5"]` instead of `["Borders:Prior=0.5"]`.

Everything else — 64 rows, two float columns, one cardinality-6 cat column,
`combinations_ctr=[]`, `max_ctr_complexity=1`, `border_count=15`, `one_hot_max_size=1`,
`permutation_count=1`, `bootstrap_type="No"`, `random_strength=0`,
`boost_from_average=False`, `score_function="L2"`, Gradient leaves,
`leaf_estimation_iterations=1`, `thread_count=1` and NO `task_type` (CPU oracle) — is the
gate-load-bearing recipe and is copied unchanged. Changing any of it makes the fit
device-unreachable or breaks a device invariant.

# EXACTLY ONE CTR descriptor (the discriminating guard)

`ECtrType::target_border_count(BinarizedTargetMeanValue, _) == 1`
(`cb-train/src/ctr/mod.rs:137-146`, mirroring `GetTargetBorderCount`,
`ctr_helper.h:34-42`): BTMV does not binarize the target at all. So one
`(projection, prior)` pair yields exactly ONE CTR column — unlike `Buckets`, which yields
one per target class. This generator therefore asserts `len(ctrs) == 1`; it must NOT
expect two, and a second descriptor means upstream emitted something other than the
intended simple BTMV column.

# The prior must lie in [0, 1] (R-3's residual hazard)

BTMV's quantizer applies `calc_normalization(prior)`, which is the identity `(0.0, 1.0)`
exactly on `[0, 1]` (DCTR-04/DCTR-05). `Prior=0.5` keeps this fixture INERT under Track
E's CPU correction — the whole point of DCTR-05 — instead of depending on it. The
assertion below pins that; never raise the prior above 1 or below 0 here.

# Seed search (mandatory, recorded)

`SEARCH_SEEDS` is scanned in order and the FIRST seed whose trained model satisfies every
anti-false-pass guard is frozen; the winning seed is recorded in `config.json` as
`data_seed`. If no seed passes, WIDEN the search (more seeds, then a higher cardinality) —
never lower a guard.

# The two HARD gate constraints (inherited from `ctr_device_mixed/`, unchanged)

  - `ctr_covered` (`session.rs`) requires `col.borders.len() + 1 == n_bins` for every CTR
    column, where `n_bins` is the device histogram width derived from the FLOAT
    quantization. `ctr_border_count_default() == 15` ⇒ 16 CTR buckets ⇒ the float
    `border_count` MUST be 15 (16 bins) or the CTR arm declines.
  - The float borders fed to the Rust trainer must be the FULL quantization set, not
    model.json's pruned USED subset — frozen here as `borders.npy`.

# Reproducibility caveat

CatBoost quantization is run-to-run nondeterministic on categorical routing; this is why
every CTR fixture in this repo is FROZEN and never regenerated in CI. Re-running this
script may produce different artifacts and would invalidate the <=1e-5 gate for every
downstream test.

# Fixed-point overflow margin (SPEC §9, mandatory)

Logloss der1 is in (-1, 1) and weights are uniform 1.0, so
`n * max(w) * max(|der1|) <= 64 * 1.0 * 1.0 = 64` << 2^33 ~= 8.6e9 — margin > 1.3e8x.
"""

import json
import os
import subprocess
import sys

import numpy as np
from catboost import CatBoost, Pool

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.dirname(HERE)
SCENARIO = "ctr_device_btmv"

N_ROWS = 64
N_FLOAT = 2
CARDS = (6,)
SEARCH_SEEDS = tuple(range(24))

CTR_PRIOR = 0.5
assert 0.0 <= CTR_PRIOR <= 1.0, (
    "the BTMV prior MUST lie in [0, 1]: calc_normalization is the identity only there, "
    "and DCTR-05 requires this fixture to be INERT under the Track E CPU correction"
)

PARAMS = {
    "loss_function": "Logloss",
    "iterations": 5,
    "depth": 2,
    "learning_rate": 0.1,
    "l2_leaf_reg": 3.0,
    "border_count": 15,
    "boosting_type": "Plain",
    "one_hot_max_size": 1,
    "max_ctr_complexity": 1,
    "simple_ctr": [f"BinarizedTargetMeanValue:Prior={CTR_PRIOR}"],
    "combinations_ctr": [],
    "permutation_count": 1,
    "fold_len_multiplier": 2.0,
    "counter_calc_method": "SkipTest",
    "leaf_estimation_method": "Gradient",
    "leaf_estimation_iterations": 1,
    "bootstrap_type": "No",
    "random_strength": 0,
    "score_function": "L2",
    "boost_from_average": False,
    "random_seed": 0,
    "thread_count": 1,
    "verbose": False,
}


def make_data(seed):
    rng = np.random.RandomState(seed)
    x = rng.rand(N_ROWS, N_FLOAT).astype(np.float32)
    cat = np.stack(
        [rng.randint(0, c, size=N_ROWS).astype(np.int32) for c in CARDS], axis=1
    )
    # Per-object float RAMP in the target (R-14): a purely categorical +/-1 target would
    # let a structure-vs-averaging leaf swap hide, because every object sharing a cat
    # value would carry the same label signal. The float term breaks that degeneracy.
    cat_term = np.isin(cat[:, 0], [0, 2, 4]).astype(np.float64)
    logit = 3.0 * (cat_term - 0.5) + 2.0 * (x[:, 0] - x[:, 1])
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.rand(N_ROWS) < prob).astype(np.int32)
    return x, cat, y


def make_pool(x, cat, y):
    df = [
        [int(cat[i, 0]), float(x[i, 0]), float(x[i, 1])] for i in range(N_ROWS)
    ]
    return Pool(df, label=y, cat_features=[0])


def fit(params, x, cat, y, pool=None):
    model = CatBoost(params)
    model.fit(pool if pool is not None else make_pool(x, cat, y))
    preds = model.predict(make_pool(x, cat, y), prediction_type="RawFormulaVal")
    return model, preds


def dump_model_json(model, tag):
    if hasattr(model, "dumps_model"):
        return json.loads(model.dumps_model())
    tmp = os.path.join(HERE, f".probe_{tag}.json")
    model.save_model(tmp, format="json")
    with open(tmp) as fh:
        model_json = json.load(fh)
    os.remove(tmp)
    return model_json


def evaluate_seed(seed):
    """Return (x, cat, y) or None if the seed's fit is unusable — degenerate target,
    degenerate predictions, no BTMV descriptor, more than one descriptor, or no float
    split."""
    x, cat, y = make_data(seed)
    if len(np.unique(y)) < 2:
        return None
    model, preds = fit(PARAMS, x, cat, y)
    if preds.std() <= 1e-6:
        return None
    model_json = dump_model_json(model, f"seed{seed}")
    ctrs = model_json["features_info"].get("ctrs", [])
    if len(ctrs) != 1:
        return None
    if ctrs[0].get("ctr_type") != "BinarizedTargetMeanValue":
        return None
    has_float_split = any(
        len(f.get("borders", [])) > 0
        for f in model_json["features_info"]["float_features"]
    )
    if not has_float_split:
        return None
    return x, cat, y


def main():
    chosen = None
    tried = []
    for seed in SEARCH_SEEDS:
        result = evaluate_seed(seed)
        tried.append(seed)
        if result is not None:
            chosen = (seed, result)
            break
    if chosen is None:
        sys.exit(
            f"no seed in {SEARCH_SEEDS} produced BOTH exactly one BinarizedTargetMeanValue "
            "CTR descriptor and a float split — widen SEARCH_SEEDS or the cardinality; do "
            "NOT lower a guard"
        )
    seed, (x, cat, y) = chosen
    print(f"seed search: scanned {tried}, chose {seed}")

    # Freeze the FULL float border set. Pool column order is [cat0, f0, f1];
    # `save_quantization_borders` indexes by the pool's FLOAT feature order.
    qpool = make_pool(x, cat, y)
    qpool.quantize(border_count=PARAMS["border_count"])
    borders_tsv = os.path.join(HERE, "borders.tsv")
    qpool.save_quantization_borders(borders_tsv)
    per_feature = {}
    with open(borders_tsv) as fh:
        for line in fh:
            fi, bv = line.split()
            per_feature.setdefault(int(fi), []).append(float(bv))
    os.remove(borders_tsv)
    float_ids = sorted(per_feature)
    assert len(float_ids) == N_FLOAT, f"expected {N_FLOAT} float border rows, got {float_ids}"
    for fi in float_ids:
        assert len(per_feature[fi]) == PARAMS["border_count"], (
            f"feature {fi}: expected {PARAMS['border_count']} borders, got {len(per_feature[fi])}"
        )
    borders = np.array([sorted(per_feature[fi]) for fi in float_ids], dtype=np.float64)

    model, predictions = fit(PARAMS, x, cat, y, pool=qpool)

    # The quantized-pool fit must be bit-identical to the raw-pool fit, or the frozen
    # borders would not describe the committed predictions.
    _raw_model, raw_preds = fit(PARAMS, x, cat, y)
    assert np.abs(predictions - raw_preds).max() == 0.0, (
        "quantized-pool fit diverged from raw-pool fit — border freezing is unsound here"
    )

    model_json_path = os.path.join(HERE, "model.json")
    model.save_model(model_json_path, format="json")
    with open(model_json_path) as fh:
        model_json = json.load(fh)

    # --- MANDATORY anti-false-pass guards -------------------------------------
    ctrs = model_json["features_info"].get("ctrs", [])
    # 1. THE discriminating assertion: EXACTLY ONE descriptor, and it is BTMV.
    #    target_border_count(BTMV) == 1 (`ctr/mod.rs:137-146`) ⇒ one column per prior.
    #    Expecting two (the Buckets shape) would be wrong for this type.
    assert len(ctrs) == 1, (
        f"expected exactly ONE CTR descriptor for a single simple BTMV prior, got "
        f"{len(ctrs)} — target_border_count(BinarizedTargetMeanValue) == 1"
    )
    ctr = ctrs[0]
    assert ctr["ctr_type"] == "BinarizedTargetMeanValue", (
        f"the single descriptor is {ctr['ctr_type']!r}, not BinarizedTargetMeanValue"
    )
    assert ctr["target_border_idx"] == 0, (
        f"BTMV never binarizes the target; target_border_idx must be 0, got "
        f"{ctr['target_border_idx']}"
    )
    assert float(ctr["prior_numerator"]) == CTR_PRIOR, (
        f"prior numerator {ctr['prior_numerator']} != pinned {CTR_PRIOR}"
    )
    assert float(ctr["prior_denomerator"]) == 1.0, (
        f"prior denominator {ctr['prior_denomerator']} != 1 (DCTR-02 pins denom == 1.0)"
    )
    assert len(ctr.get("elements", [])) == 1, (
        "max_ctr_complexity=1 ⇒ the projection must be simple (one member)"
    )
    # 2. >=1 float split, or the device n_features arithmetic is untested.
    assert any(
        len(f.get("borders", [])) > 0
        for f in model_json["features_info"]["float_features"]
    ), "no float split in the model — the float axis is decorative"
    # 3. Non-degenerate predictions.
    assert predictions.std() > 1e-6, "degenerate constant predictions"

    observed_target_border_idxs = sorted({int(c["target_border_idx"]) for c in ctrs})
    assert observed_target_border_idxs == [0], (
        f"BTMV exposes only b = 0, observed {observed_target_border_idxs}"
    )

    np.save(os.path.join(HERE, "borders.npy"), borders)
    np.save(os.path.join(HERE, "X.npy"), x)
    # ONE cat column ⇒ a 1-D [N] int32 column, matching `ctr_device_mixed/X_cat.npy`
    # and the `[N] int32` load the DCTR-14 end-to-end test performs.
    np.save(os.path.join(HERE, "X_cat.npy"), cat[:, 0])
    np.save(os.path.join(HERE, "y.npy"), y.astype(np.float64))
    np.save(os.path.join(HERE, "predictions.npy"), predictions.astype(np.float64))

    config = {
        "scenario": SCENARIO,
        "requirement": "DCTR-14",
        "data_seed": seed,
        "seed_search": list(SEARCH_SEEDS),
        "seeds_tried": tried,
        "escalation_rung": 0,
        "catboost_version": "1.2.10",
        "thread_count": 1,
        "n_rows": N_ROWS,
        "cardinalities": list(CARDS),
        "ctr_prior": CTR_PRIOR,
        "n_ctr_descriptors": len(ctrs),
        "observed_target_border_idxs": observed_target_border_idxs,
        "overflow_margin": (
            "n*max(w)*max(|der1|) <= 64*1.0*1.0 = 64 << 2^33 ~= 8.6e9 (margin > 1.3e8x) "
            "— Logloss der1 is bounded by 1 in magnitude"
        ),
        "description": (
            "BinarizedTargetMeanValue-CTR device-parity fixture (DCTR-14). 64 rows, two "
            "float columns (border_count=15, FULL border set frozen as borders.npy) plus "
            "ONE cardinality-6 cat column routed to CTR (one_hot_max_size=1), "
            "simple_ctr=BinarizedTargetMeanValue:Prior=0.5, combinations_ctr=[], "
            "max_ctr_complexity=1. Unlike ctr_btmv_simple/ (categorical-only, hence "
            "device-unreachable) this pool carries float columns, so the fit can commit "
            "to the device. target_border_count(BTMV) == 1, so upstream emits EXACTLY ONE "
            "CTR descriptor for the single prior — asserted, and the smoke test pins it. "
            "The prior is in [0,1], so calc_normalization is the identity and this fixture "
            "is INERT under the Track E CPU correction (DCTR-05). Pins the f32 "
            "TCtrMeanHistory::Sum accumulator semantics the device BTMV path must "
            "reproduce at <=1e-5. NEVER regenerated in CI."
        ),
        "params": PARAMS,
        "npy_schema": {
            "borders.npy": "[2,15] f64 — FULL frozen float border set (feed to train_cat)",
            "X.npy": "[N,2] f32 — float feature matrix",
            "X_cat.npy": "[N] int32 — categorical codes (stringified A4 form on the Rust side)",
            "y.npy": "[N] f64 — binclf label",
            "predictions.npy": "[N] f64 — RawFormulaVal (<=1e-5 gate)",
        },
        "note": (
            "FROZEN. Generated once by this directory's own gen_fixtures.py and NEVER "
            "regenerated in CI. CatBoost quantization is run-to-run nondeterministic on "
            "categorical routing; regenerating invalidates the <=1e-5 gate."
        ),
    }
    with open(os.path.join(HERE, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)
        fh.write("\n")

    out = subprocess.run(
        ["git", "status", "--porcelain", FIXTURES],
        capture_output=True,
        text=True,
        cwd=FIXTURES,
    ).stdout
    offenders = [line for line in out.splitlines() if line.strip() and SCENARIO not in line]
    if offenders:
        print("corpus contamination — this generator touched paths outside its scenario:")
        for line in offenders:
            print("   ", line)
        sys.exit(1)

    n_ctr_splits = sum(len(c.get("borders", [])) for c in ctrs)
    print(
        f"wrote {SCENARIO}: seed={seed}, ctr descriptors={len(ctrs)} "
        f"(used ctr borders={n_ctr_splits}), predictions.std()={predictions.std():.6f}"
    )


if __name__ == "__main__":
    main()
