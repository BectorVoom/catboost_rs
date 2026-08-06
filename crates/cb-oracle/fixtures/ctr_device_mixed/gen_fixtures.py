"""Generate the MIXED float+categorical CTR device-parity fixture (GDC-06/T13, for GDC-12/T15).

FROZEN GENERATOR — the committed artifacts under this directory are the GROUND
TRUTH the device CTR arm (structure-permutation split search + AVERAGING-
permutation leaf-value gather, GDC-09/10/11) is compared against at <=1e-5. CI
does NOT run this script and never regenerates these fixtures.

# Why this fixture exists (V-9)

Every existing CTR e2e fixture is categorical-ONLY (deliberate float-border
nondeterminism mitigation), but a cat-only pool yields `device_n_features == 0` /
`device_n_bins == 0` (`boosting.rs`), so it can NEVER commit to the device even
after the GDC-11 clause relaxation; `plain_ctr/` has no trained model at all.
This fixture is the first CTR fixture with float columns, so the device CTR arm
is actually reachable.

# The two HARD gate constraints (verified, PLAN V-9)

  - `ctr_covered` (session.rs) requires `col.borders.len() + 1 == n_bins` for
    every CTR column, where `n_bins` is the device histogram width derived from
    the FLOAT quantization. `ctr_border_count_default() == 15` ⇒ 16 CTR buckets ⇒
    the float `border_count` MUST be 15 (16 bins) or the CTR arm declines.
  - The float borders fed to the Rust trainer must be the FULL quantization set,
    not model.json's pruned USED subset — frozen here as `borders.npy` via
    `Pool.quantize()` + `save_quantization_borders` (fit on the quantized pool is
    asserted bit-identical to the raw fit below).

# Permutation-divergence guard (SPEC GDC-12 Unresolved)

The structure CTR order is the learn-set shuffle `S` and the averaging CTR order
is `S ∘ P_avg` (two independent draws off one seeded stream) — they cannot agree
at n=64 unless the permutation code itself degenerates. The load-bearing form of
this guard is asserted RUST-side, where the real permutations are computable:
T11's session self-oracle asserts the averaging CTR COLUMNS differ from the
structure columns, and T15's e2e test asserts the two permutations differ for
this fixture's exact (n, seed). Python cannot reproduce the Rust/upstream RNG
stream, so no permutation .npy files are emitted here.

# Pinned recipe

  - `catboost==1.2.10`, `numpy.random.RandomState(5)` (data; training seed 0), `thread_count=1`.
  - 64 rows, TWO float columns in [0, 1) (`border_count=15`), ONE categorical
    column of cardinality 6.
  - `one_hot_max_size=1` routes the cat column to CTR (not one-hot).
  - `simple_ctr=["Borders:Prior=0.5"]`, `combinations_ctr=[]`,
    `max_ctr_complexity=1`, `counter_calc_method="SkipTest"` (CtrBorderCount
    stays at upstream's default 15 = 16 CTR buckets, matching
    `ctr_border_count_default()`).
  - Logloss, iterations=5, depth=2, lr=0.1, l2=3.0, Plain, permutation_count=1,
    fold_len_multiplier=2.0, bootstrap No, random_strength 0,
    boost_from_average False, Gradient leaf, score_function L2.
  - The target depends on BOTH the cat column and the float columns so the model
    genuinely selects BOTH split kinds (asserted below).
"""

import json
import os
import subprocess
import sys

import numpy as np
from catboost import CatBoost, Pool

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.dirname(HERE)
SCENARIO = "ctr_device_mixed"

N_ROWS = 64
N_FLOAT = 2
CARD = 6

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
    "simple_ctr": ["Borders:Prior=0.5"],
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


def main():
    # Data seed 5 (training random_seed stays 0): scanned seeds 0..5 — this is the
    # first (seed, coefficient) combination where the trained model uses BOTH a
    # CTR split and float splits (guards 1 and 2 below).
    rng = np.random.RandomState(5)

    x = rng.rand(N_ROWS, N_FLOAT).astype(np.float32)
    cat = rng.randint(0, CARD, size=N_ROWS).astype(np.int32)

    # BOTH-driven target: cat AND float signal, so upstream selects both split
    # kinds (guards 1 and 2 below assert that outcome away from vacuity).
    logit = 1.5 * np.isin(cat, [0, 2, 4]).astype(np.float64) + 2.5 * (x[:, 0] - x[:, 1])
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.rand(N_ROWS) < prob).astype(np.int32)
    if len(np.unique(y)) < 2:
        sys.exit("degenerate target: only one class present")

    def make_pool():
        df = [[int(cat[i]), float(x[i, 0]), float(x[i, 1])] for i in range(N_ROWS)]
        return Pool(df, label=y, cat_features=[0])

    # Freeze the FULL float border set (see module docstring). Column order in the
    # pool is [cat, f0, f1]; `save_quantization_borders` indexes by the pool's
    # FLOAT feature order, i.e. flat feature indices 1 and 2.
    qpool = make_pool()
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

    model = CatBoost(PARAMS)
    model.fit(qpool)
    predictions = model.predict(make_pool(), prediction_type="RawFormulaVal")

    # The quantized-pool fit must be bit-identical to the raw-pool fit, or the
    # frozen borders would not describe the committed predictions.
    raw_model = CatBoost(PARAMS)
    raw_model.fit(make_pool())
    raw_preds = raw_model.predict(make_pool(), prediction_type="RawFormulaVal")
    assert np.abs(predictions - raw_preds).max() == 0.0, (
        "quantized-pool fit diverged from raw-pool fit — border freezing is unsound here"
    )

    model_json_path = os.path.join(HERE, "model.json")
    model.save_model(model_json_path, format="json")
    with open(model_json_path) as fh:
        model_json = json.load(fh)

    # --- MANDATORY anti-false-pass guards (PLAN T13, all three) ---------------
    # 1. >=1 CTR split (else the CTR arm is decorative).
    ctrs = model_json["features_info"].get("ctrs", [])
    assert any(c["ctr_type"] == "Borders" for c in ctrs), (
        "no Borders CTR descriptor in model.json — zero CTR splits, fixture is vacuous"
    )
    # 2. >=1 float split (else the float axis is decorative and the device
    #    n_features arithmetic would be untested). model.json borders are the
    #    USED subset, so non-empty == used-by-a-split.
    float_features = model_json["features_info"]["float_features"]
    assert any(len(f["borders"]) > 0 for f in float_features), (
        "no float split in the model — the float axis is decorative"
    )
    # 3. Permutation divergence: asserted RUST-side (see module docstring).
    assert predictions.std() > 1e-6, "degenerate constant predictions"

    np.save(os.path.join(HERE, "borders.npy"), borders)
    np.save(os.path.join(HERE, "X.npy"), x)
    np.save(os.path.join(HERE, "X_cat.npy"), cat)
    np.save(os.path.join(HERE, "y.npy"), y.astype(np.float64))
    np.save(os.path.join(HERE, "predictions.npy"), predictions.astype(np.float64))

    config = {
        "scenario": SCENARIO,
        "requirement": "GDC-06 / GDC-12",
        "seed": 5,
        "catboost_version": "1.2.10",
        "thread_count": 1,
        "n_rows": N_ROWS,
        "cardinality": CARD,
        "description": (
            "Mixed float+categorical CTR device-parity fixture. 64 rows, two float "
            "columns (border_count=15, FULL border set frozen as borders.npy) plus one "
            "cardinality-6 cat column routed to a simple Borders CTR "
            "(one_hot_max_size=1, Prior=0.5, CtrBorderCount default 15 = 16 buckets "
            "matching the 16 float bins — the ctr_covered n_bins requirement). Logloss "
            "depth-2 x5, Plain, permutation_count=1, no sampling, bias 0, Gradient, L2 "
            "score. The target is driven by BOTH split kinds (asserted). Pins the "
            "two-permutation CTR semantics (structure split search + averaging leaf "
            "values) the device CTR arm must reproduce at <=1e-5. NEVER regenerated in CI."
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
            "FROZEN. Generated once by this directory's own gen_fixtures.py and "
            "NEVER regenerated in CI. Regenerating invalidates the <=1e-5 gate."
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

    n_ctr_splits = sum(len(f.get("borders", [])) for f in ctrs)
    print(
        f"wrote {SCENARIO}: {N_ROWS} rows, ctr descriptors={len(ctrs)} "
        f"(used ctr borders={n_ctr_splits}), predictions.std()={predictions.std():.6f}"
    )


if __name__ == "__main__":
    main()
