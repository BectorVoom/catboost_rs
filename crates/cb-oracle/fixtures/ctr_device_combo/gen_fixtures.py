"""Generate the COMBINATION-CTR device-parity fixture (FPP-10/T03).

FROZEN GENERATOR — the committed artifacts under this directory are the GROUND TRUTH
the device combination/tensor CTR column builder (FPP-09,
`combine_projection_bins` + per-member `member_bins`) is compared against at <=1e-5.
CI does NOT run this script and never regenerates these fixtures.

# Why this fixture exists (PLAN V-4 / T03)

`tensor_ctr_e2e/` and `ctr_mixed_simple_vs_combo/` both DO contain multi-member CTR
projections, but they are categorical-ONLY pools. A cat-only pool gives
`device_n_float == matrix.n_features() == 0` (`boosting.rs`), so
`has_any_scorable_feature` and the session's own `n_features == 0` decline both fire
— those fixtures can never reach the device. `ctr_device_mixed/` IS device-reachable
but has exactly ONE cat column, so no combination is possible.

This is the first fixture that is BOTH device-reachable (float columns present) and
carries a >=2-member CTR projection.

# The two HARD gate constraints (inherited from `ctr_device_mixed/`, unchanged)

  - `ctr_covered` (`session.rs`) requires `col.borders.len() + 1 == n_bins` for every
    CTR column, where `n_bins` is the device histogram width derived from the FLOAT
    quantization. `ctr_border_count_default() == 15` ⇒ 16 CTR buckets ⇒ the float
    `border_count` MUST be 15 (16 bins) or the CTR arm declines.
  - The float borders fed to the Rust trainer must be the FULL quantization set, not
    model.json's pruned USED subset — frozen here as `borders.npy`.

# Deltas from `ctr_device_mixed/` (the proven device-reachable CTR recipe)

  - TWO cat columns (small cardinalities) instead of one.
  - `max_ctr_complexity=2` instead of 1 — admits the 2-member projection.
  - `combinations_ctr=["Borders:Prior=0.5"]` instead of `[]` — without this upstream
    has no descriptor to build a combination CTR from.
  - `one_hot_max_size=1` (unchanged) forces BOTH cat columns onto the CTR route.

Everything else is copied verbatim.

# Seed search (mandatory, recorded per T03)

Upstream is free to pick only simple projections even with `max_ctr_complexity=2` —
whether a combination split wins is data-dependent. `SEARCH_SEEDS` below is scanned in
order and the FIRST seed whose trained model contains a >=2-member CTR projection AND
at least one float split is frozen. The winning seed is recorded in `config.json` as
`data_seed`. If the assertion at the end fires, widen the search rather than lowering
the bar — a fixture without a combination projection cannot exercise FPP-09.

# Reproducibility caveat

CatBoost quantization is run-to-run nondeterministic on categorical routing; this is
why every CTR fixture in this repo is FROZEN and never regenerated in CI.

# Fixed-point overflow margin (SPEC §9, mandatory)

Logloss der1 ∈ (-1, 1) and weights are uniform 1.0, so
`n · max(w) · max(|der1|) <= 64 · 1.0 · 1.0 = 64` << 2^33 ≈ 8.6e9 — margin > 1.3e8x.
"""

import json
import os
import subprocess
import sys

import numpy as np
from catboost import CatBoost, Pool

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.dirname(HERE)
SCENARIO = "ctr_device_combo"

N_ROWS = 64
N_FLOAT = 2
CARDS = (3, 4)
SEARCH_SEEDS = tuple(range(24))

PARAMS = {
    "loss_function": "Logloss",
    "iterations": 5,
    "depth": 2,
    "learning_rate": 0.1,
    "l2_leaf_reg": 3.0,
    "border_count": 15,
    "boosting_type": "Plain",
    "one_hot_max_size": 1,
    "max_ctr_complexity": 2,
    "simple_ctr": ["Borders:Prior=0.5"],
    "combinations_ctr": ["Borders:Prior=0.5"],
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


def projection_len(ctr):
    """Members in a CTR descriptor's projection — `elements` has one entry per member
    (verified against `tensor_ctr_e2e/model.json`)."""
    return len(ctr.get("elements", []))


def make_data(seed):
    rng = np.random.RandomState(seed)
    x = rng.rand(N_ROWS, N_FLOAT).astype(np.float32)
    cat = np.stack(
        [rng.randint(0, c, size=N_ROWS).astype(np.int32) for c in CARDS], axis=1
    )
    # INTERACTION-driven target: the logit depends on the PAIR (cat0, cat1) via an XOR-ish
    # term that neither column explains alone, so a 2-member combination CTR is genuinely
    # the better split — plus a float term so the float axis is not decorative.
    pair = ((cat[:, 0] % 2) ^ (cat[:, 1] % 2)).astype(np.float64)
    logit = 3.0 * (pair - 0.5) + 2.0 * (x[:, 0] - x[:, 1])
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.rand(N_ROWS) < prob).astype(np.int32)
    return x, cat, y


def make_pool(x, cat, y):
    df = [
        [int(cat[i, 0]), int(cat[i, 1]), float(x[i, 0]), float(x[i, 1])]
        for i in range(N_ROWS)
    ]
    return Pool(df, label=y, cat_features=[0, 1])


def fit(params, x, cat, y, pool=None):
    model = CatBoost(params)
    model.fit(pool if pool is not None else make_pool(x, cat, y))
    preds = model.predict(make_pool(x, cat, y), prediction_type="RawFormulaVal")
    return model, preds


def evaluate_seed(seed):
    """Return (x, cat, y, model, preds, model_json, max_members) or None if the seed's
    fit is unusable (degenerate target, no combination projection, or no float split)."""
    x, cat, y = make_data(seed)
    if len(np.unique(y)) < 2:
        return None
    model, preds = fit(PARAMS, x, cat, y)
    if preds.std() <= 1e-6:
        return None
    model_json = json.loads(model.dumps_model()) if hasattr(model, "dumps_model") else None
    if model_json is None:
        tmp = os.path.join(HERE, f".probe_seed{seed}.json")
        model.save_model(tmp, format="json")
        with open(tmp) as fh:
            model_json = json.load(fh)
        os.remove(tmp)
    ctrs = model_json["features_info"].get("ctrs", [])
    max_members = max((projection_len(c) for c in ctrs), default=0)
    has_float_split = any(
        len(f.get("borders", [])) > 0
        for f in model_json["features_info"]["float_features"]
    )
    if max_members < 2 or not has_float_split:
        return None
    return x, cat, y, model, preds, model_json, max_members


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
            f"no seed in {SEARCH_SEEDS} produced BOTH a >=2-member CTR projection and a "
            "float split — widen SEARCH_SEEDS or the cardinalities; do NOT lower the bar"
        )
    seed, (x, cat, y, _model, _preds, _mj, max_members) = chosen
    print(f"seed search: scanned {tried}, chose {seed} (max projection members={max_members})")

    # Freeze the FULL float border set. Pool column order is [cat0, cat1, f0, f1];
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
    max_members = max((projection_len(c) for c in ctrs), default=0)
    # 1. THE discriminating assertion: a >=2-member projection must be present.
    assert max_members >= 2, (
        f"widest CTR projection has {max_members} member(s) after quantization — the "
        "combination-CTR column builder would be unexercised"
    )
    # 2. >=1 float split, or the device n_features arithmetic is untested.
    assert any(
        len(f.get("borders", [])) > 0
        for f in model_json["features_info"]["float_features"]
    ), "no float split in the model — the float axis is decorative"
    # 3. The combination must be load-bearing: a max_ctr_complexity=1 sibling must
    #    predict differently, or `max_ctr_complexity` is not actually doing anything.
    _c1_model, c1_preds = fit(
        dict(PARAMS, max_ctr_complexity=1, combinations_ctr=[]), x, cat, y
    )
    combo_delta = float(np.abs(predictions - c1_preds).max())
    assert combo_delta > 1e-6, (
        f"max_ctr_complexity=2 and =1 predictions agree (max|Δ|={combo_delta:.3e}) — "
        "the fixture is vacuous"
    )
    assert predictions.std() > 1e-6, "degenerate constant predictions"

    np.save(os.path.join(HERE, "borders.npy"), borders)
    np.save(os.path.join(HERE, "X.npy"), x)
    np.save(os.path.join(HERE, "X_cat.npy"), cat)
    np.save(os.path.join(HERE, "y.npy"), y.astype(np.float64))
    np.save(os.path.join(HERE, "predictions.npy"), predictions.astype(np.float64))

    config = {
        "scenario": SCENARIO,
        "requirement": "FPP-10 / FPP-11",
        "data_seed": seed,
        "seed_search": list(SEARCH_SEEDS),
        "seeds_tried": tried,
        "catboost_version": "1.2.10",
        "thread_count": 1,
        "n_rows": N_ROWS,
        "cardinalities": list(CARDS),
        "max_projection_members": max_members,
        "combo_vs_complexity1_max_delta": combo_delta,
        "overflow_margin": (
            "n*max(w)*max(|der1|) <= 64*1.0*1.0 = 64 << 2^33 ~= 8.6e9 (margin > 1.3e8x) "
            "— Logloss der1 is bounded by 1 in magnitude"
        ),
        "description": (
            "Combination/tensor-CTR device-parity fixture. 64 rows, two float columns "
            "(border_count=15, FULL border set frozen as borders.npy) plus TWO small-"
            "cardinality cat columns routed to CTR (one_hot_max_size=1), "
            "max_ctr_complexity=2 with combinations_ctr=Borders:Prior=0.5. The target's "
            "logit carries an XOR interaction between the two cat columns that neither "
            "explains alone, so upstream genuinely selects a 2-member combination "
            "projection (asserted). Pins the combined-bin semantics the device "
            "`combine_projection_bins` path (FPP-09) must reproduce at <=1e-5. NEVER "
            "regenerated in CI."
        ),
        "params": PARAMS,
        "npy_schema": {
            "borders.npy": "[2,15] f64 — FULL frozen float border set (feed to train_cat)",
            "X.npy": "[N,2] f32 — float feature matrix",
            "X_cat.npy": "[N,2] int32 — categorical codes (stringified A4 form on the Rust side)",
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

    print(
        f"wrote {SCENARIO}: seed={seed}, ctr descriptors={len(ctrs)}, "
        f"max projection members={max_members}, combo-vs-complexity1 max|Δ|={combo_delta:.6f}"
    )


if __name__ == "__main__":
    main()
