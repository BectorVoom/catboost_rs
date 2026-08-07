"""Generate the WEIGHTED-training device-parity fixture, Depthwise (non-symmetric) arm (GDC-06/T03).

FROZEN GENERATOR — the committed `X.npy` / `y.npy` / `weights.npy` / `model.json` /
`predictions.npy` / `config.json` under this directory are the GROUND TRUTH the
device weighted-derivative path (GDC-02/GDC-05, `Σ(w·der1)/(Σw+l2)`) is compared
against at <=1e-5. CI does NOT run this script (no `catboost` install in CI) and
never regenerates these fixtures.

# Reproducibility caveat (load-bearing — do not "fix" by re-running this file)

CatBoost's float-border quantization has run-to-run nondeterminism independent of
`random_seed` (see `crates/cb-oracle/fixtures/ctr_load/gen_fixtures.py`), and the
saved `model.json` keeps only the borders the trained model USED (verified:
`ordered_boost_e2e/model.json` has [2, 0] borders), which is too few for the
device gate's `n_bins` arithmetic. This fixture therefore freezes the FULL
quantization border set explicitly: the pool is quantized ONCE
(`Pool.quantize(border_count=15)`), the borders are exported via
`save_quantization_borders` and committed as `borders.npy` `[2, 15]`, and the
model is trained on that exact quantized pool (verified bit-identical to the
raw-pool fit). The Rust side feeds `borders.npy` to `train`, so both sides
quantize identically by construction. The committed artifacts ARE the fixtures.

# Why this fixture exists

No existing fixture trains a model end-to-end with NON-UNIFORM object weights
(`class_weights/` is only a raw weight array). The device grow path summed the
UNWEIGHTED der (`Σ der1`) into histogram channel 0 (WR-03), which agrees with a
uniform-weight fit but NOT with upstream's `Σ w·der` on a genuinely weighted pool.
This fixture pins the weighted formula end to end against real upstream CatBoost.

# Fixed-point overflow margin (SPEC §9, mandatory)

The device fixed-point histogram requires |Σ| < 2^33 ≈ 8.6e9 (kernels.rs). Here
`n · max(w) · max(|der1|) <= 64 · 3.0 · max|y| <= 64 · 3.0 · 10 = 1920`
(|y| is bounded ≈ 5.6 by construction below, and the RMSE der never exceeds the
target range on a bounded fit), a margin > 4.4e6× under the bound.

# Pinned recipe

  - `catboost==1.2.10`, `numpy.random.RandomState(0)`, `thread_count=1`.
  - 64 rows, TWO float columns in [0, 1), ZERO categorical columns.
  - `border_count=15` (16 bins), RMSE, iterations=3, depth=3, lr=0.3, l2=3.0.
  - Plain / bootstrap No / random_strength 0 / boost_from_average False (bias 0,
    the CR-01 device convention) / Gradient leaf / score_function L2.
  - `grow_policy=Depthwise`.
  - `sample_weight` NON-uniform: w[i] cycles {0.5, 1.0, 2.0, 3.0}.
"""

import json
import os
import subprocess
import sys

import numpy as np
from catboost import CatBoost, Pool

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.dirname(HERE)
SCENARIO = "weighted_device_nonsym"

N_ROWS = 64
N_FLOAT = 2
WEIGHT_CYCLE = [0.5, 1.0, 2.0, 3.0]

PARAMS = {
    "loss_function": "RMSE",
    "iterations": 3,
    "depth": 3,
    "learning_rate": 0.3,
    "l2_leaf_reg": 3.0,
    "border_count": 15,
    "boosting_type": "Plain",
    "permutation_count": 1,
    "bootstrap_type": "No",
    "random_strength": 0,
    "boost_from_average": False,
    "leaf_estimation_method": "Gradient",
    "leaf_estimation_iterations": 1,
    "score_function": "L2",
    "grow_policy": "Depthwise",
    "random_seed": 0,
    "thread_count": 1,
    "verbose": False,
}


def main():
    rng = np.random.RandomState(0)

    x = rng.rand(N_ROWS, N_FLOAT).astype(np.float32)
    # Bounded target: |y| <= 3 + 2 + 0.5 + noise(0.5) < 10 by construction — the
    # overflow-margin arithmetic in the module docstring depends on this bound.
    y = (
        3.0 * x[:, 0]
        - 2.0 * x[:, 1]
        + 0.5 * np.sin(6.0 * x[:, 0])
        + 0.5 * rng.randn(N_ROWS)
    ).astype(np.float64)
    assert np.abs(y).max() < 10.0, "target bound backs the 2^33 margin arithmetic"

    weights = np.array(
        [WEIGHT_CYCLE[i % len(WEIGHT_CYCLE)] for i in range(N_ROWS)], dtype=np.float64
    )

    # Quantize ONCE and freeze the FULL border set (see the module docstring):
    # the fit on the quantized pool is bit-identical to the raw-pool fit, and
    # `borders.npy` is what the Rust side trains with.
    pool = Pool(x, label=y, weight=weights)
    pool.quantize(border_count=PARAMS["border_count"])
    borders_tsv = os.path.join(HERE, "borders.tsv")
    pool.save_quantization_borders(borders_tsv)
    per_feature = {}
    with open(borders_tsv) as fh:
        for line in fh:
            fi, bv = line.split()
            per_feature.setdefault(int(fi), []).append(float(bv))
    os.remove(borders_tsv)
    assert sorted(per_feature) == list(range(N_FLOAT))
    for fi, bs in per_feature.items():
        assert len(bs) == PARAMS["border_count"], (
            f"feature {fi}: expected {PARAMS['border_count']} borders, got {len(bs)}"
        )
    borders = np.array(
        [sorted(per_feature[fi]) for fi in range(N_FLOAT)], dtype=np.float64
    )

    model = CatBoost(PARAMS)
    model.fit(pool)
    eval_pool = Pool(x, label=y, weight=weights)
    predictions = model.predict(eval_pool, prediction_type="RawFormulaVal")

    # --- MANDATORY anti-false-pass guards -------------------------------------
    # 1. The weighted fit must actually differ from an unweighted fit, or this
    #    fixture cannot discriminate the weighted formula.
    unweighted = CatBoost(PARAMS)
    unweighted.fit(Pool(x, label=y))
    unweighted_preds = unweighted.predict(eval_pool, prediction_type="RawFormulaVal")
    max_delta = np.abs(predictions - unweighted_preds).max()
    assert max_delta > 1e-6, (
        f"weighted and unweighted predictions agree (max|Δ|={max_delta:.3e}) — "
        "the fixture is vacuous"
    )
    # 2. Non-degenerate model.
    assert predictions.std() > 1e-6, "degenerate constant predictions"

    model_json_path = os.path.join(HERE, "model.json")
    model.save_model(model_json_path, format="json")
    with open(model_json_path) as fh:
        model_json = json.load(fh)
    # Sanity: every border the model actually USES is present in the frozen full
    # set (the model.json set is the pruned subset).
    for fi, f in enumerate(model_json["features_info"]["float_features"]):
        for b in f["borders"]:
            assert any(abs(b - fb) < 1e-9 for fb in borders[fi]), (
                f"model border {b} (feature {fi}) missing from frozen border set"
            )

    np.save(os.path.join(HERE, "borders.npy"), borders)
    np.save(os.path.join(HERE, "X.npy"), x)
    np.save(os.path.join(HERE, "y.npy"), y)
    np.save(os.path.join(HERE, "weights.npy"), weights)
    np.save(os.path.join(HERE, "predictions.npy"), predictions.astype(np.float64))

    config = {
        "scenario": SCENARIO,
        "requirement": "GDC-06 / GDC-08",
        "seed": 0,
        "catboost_version": "1.2.10",
        "thread_count": 1,
        "n_rows": N_ROWS,
        "weight_cycle": WEIGHT_CYCLE,
        "overflow_margin": (
            "n*max(w)*max(|der1|) <= 64*3.0*10 = 1920 << 2^33 ~= 8.6e9 "
            "(margin > 4.4e6x) — the device fixed-point histogram precondition"
        ),
        "description": (
            "Weighted-training device-parity fixture, Depthwise (non-symmetric) arm. 64 rows, two "
            "float columns, border_count=15, RMSE depth-3 x3 iterations, Plain, no "
            "sampling, bias 0, Gradient leaf, L2 score, sample_weight cycling "
            "{0.5,1.0,2.0,3.0}. Pins upstream's SumWeightedDelta leaf/histogram "
            "semantics (leaf = Σ(w·der1)/(Σw+l2)) that the device weighted-der "
            "channel (GDC-02) must reproduce at <=1e-5. NEVER regenerated in CI."
        ),
        "params": PARAMS,
        "npy_schema": {
            "borders.npy": "[2,15] f64 — the FULL frozen quantization border set (feed to train)",
            "X.npy": "[N,2] f32 — float feature matrix",
            "y.npy": "[N] f64 — RMSE target, |y| < 10 by construction",
            "weights.npy": "[N] f64 — NON-uniform per-object weights",
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

    print(
        f"wrote {SCENARIO}: {N_ROWS} rows, weighted-vs-unweighted max|Δ|={max_delta:.6f}, "
        f"predictions.std()={predictions.std():.6f}"
    )


if __name__ == "__main__":
    main()
