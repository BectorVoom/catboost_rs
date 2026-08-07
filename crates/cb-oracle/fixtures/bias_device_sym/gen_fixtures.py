"""Generate the NON-ZERO-BIAS device-parity fixture, SymmetricTree arm (FPP-03/T01).

FROZEN GENERATOR — the committed `X.npy` / `y.npy` / `borders.npy` / `model.json` /
`predictions.npy` / `config.json` under this directory are the GROUND TRUTH the
device non-zero starting-approximant path (FPP-01/FPP-02, `DeviceTrainConfig.bias`)
is compared against at <=1e-5. CI does NOT run this script (no `catboost` install in
CI) and never regenerates these fixtures.

# Reproducibility caveat (load-bearing — do not "fix" by re-running this file)

CatBoost's float-border quantization has run-to-run nondeterminism independent of
`random_seed`, and the saved `model.json` keeps only the borders the trained model
USED, which is too few for the device gate's `n_bins` arithmetic. This fixture
therefore freezes the FULL quantization border set explicitly: the pool is quantized
ONCE (`Pool.quantize(border_count=15)`), the borders are exported via
`save_quantization_borders` and committed as `borders.npy` `[2, 15]`, and the model
is trained on that exact quantized pool. The Rust side feeds `borders.npy` to
`train`, so both sides quantize identically by construction.

# Why this fixture exists

EVERY existing device-eligible fixture pins `boost_from_average=False` (PLAN V-8),
because `device_host_eligible` declined any fit with a non-zero starting
approximant (`boosting.rs`'s `bias == 0.0` clause). `GpuTrainSession::begin` seeded
the resident approximant to a hardcoded `vec![0.0; n]`, so a `boost_from_average`
fit's very first resident `der1` would have been wrong. This fixture is the first to
pin an upstream fit whose starting approximant is `mean(y) != 0`.

# Why the target mean is shifted

`mean(y)` IS the starting approximant for an RMSE fit. A near-zero mean cannot
discriminate the fix from the former hardcoded-zero seed, so `y` carries a +1.5
offset giving `|mean(y)| ~ 2.0`, and the smoke test asserts `|mean(y)| > 0.5`.

# Fixed-point overflow margin (SPEC §9, mandatory)

The device fixed-point histogram requires |Σ| < 2^33 ≈ 8.6e9 (kernels.rs). Here the
weights are UNIFORM 1.0 (the bias axis is isolated from the weight axis), so
`n · max(w) · max(|der1|) <= 64 · 1.0 · max|y| <= 64 · 1.0 · 10 = 640`
(|y| is bounded < 10 by construction below, and the RMSE der1 = y - approx never
exceeds the target range on a bounded fit), a margin > 1.3e7x under the bound.

# Pinned recipe (every value load-bearing)

  - `catboost==1.2.10`, `numpy.random.RandomState(0)`, `thread_count=1`.
  - 64 rows, TWO float columns in [0, 1), ZERO categorical columns.
  - `border_count=15` (16 bins; `pad_hist_line_bins(16) = 32`, an admitted
    resident histogram line width — PLAN blocker B-3, resolved).
  - RMSE, iterations=3, depth=3, lr=0.3, l2=3.0.
  - Plain / bootstrap No / random_strength 0 / Gradient leaf / L2 score.
  - `grow_policy=SymmetricTree`.
  - `boost_from_average=True`  <- THE POINT OF THIS FIXTURE.
  - UNIFORM sample weight (isolate the bias axis from the weight axis).
"""

import json
import os
import subprocess
import sys

import numpy as np
from catboost import CatBoost, Pool

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.dirname(HERE)
SCENARIO = "bias_device_sym"

N_ROWS = 64
N_FLOAT = 2
# Shifts mean(y) — the RMSE starting approximant — well clear of zero.
TARGET_OFFSET = 1.5

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
    "boost_from_average": True,
    "leaf_estimation_method": "Gradient",
    "leaf_estimation_iterations": 1,
    "score_function": "L2",
    "grow_policy": "SymmetricTree",
    "random_seed": 0,
    "thread_count": 1,
    "verbose": False,
}


def main():
    rng = np.random.RandomState(0)

    x = rng.rand(N_ROWS, N_FLOAT).astype(np.float32)
    # Bounded target: |y| <= 1.5 + 3 + 2 + 0.5 + noise(~2) < 10 by construction — the
    # overflow-margin arithmetic in the module docstring depends on this bound.
    y = (
        TARGET_OFFSET
        + 3.0 * x[:, 0]
        - 2.0 * x[:, 1]
        + 0.5 * np.sin(6.0 * x[:, 0])
        + 0.5 * rng.randn(N_ROWS)
    ).astype(np.float64)
    assert np.abs(y).max() < 10.0, "target bound backs the 2^33 margin arithmetic"
    assert abs(y.mean()) > 0.5, (
        f"|mean(y)|={abs(y.mean()):.6f} <= 0.5 — a near-zero starting approximant "
        "cannot discriminate the bias fix from the former hardcoded-zero seed"
    )

    # Quantize ONCE and freeze the FULL border set (see the module docstring).
    pool = Pool(x, label=y)
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
    eval_pool = Pool(x, label=y)
    predictions = model.predict(eval_pool, prediction_type="RawFormulaVal")

    # --- MANDATORY anti-false-pass guards -------------------------------------
    # 1. The boost_from_average fit must actually differ from a zero-bias fit, or
    #    this fixture cannot discriminate the starting-approximant path at all.
    zero_bias_params = dict(PARAMS, boost_from_average=False)
    zero_bias = CatBoost(zero_bias_params)
    zero_bias.fit(Pool(x, label=y))
    zero_bias_preds = zero_bias.predict(eval_pool, prediction_type="RawFormulaVal")
    max_delta = np.abs(predictions - zero_bias_preds).max()
    assert max_delta > 1e-6, (
        f"boost_from_average=True and =False predictions agree (max|Δ|={max_delta:.3e}) "
        "— the fixture is vacuous"
    )
    # 2. Non-degenerate model.
    assert predictions.std() > 1e-6, "degenerate constant predictions"

    model_json_path = os.path.join(HERE, "model.json")
    model.save_model(model_json_path, format="json")
    with open(model_json_path) as fh:
        model_json = json.load(fh)
    # Sanity: every border the model actually USES is present in the frozen full set.
    for fi, f in enumerate(model_json["features_info"]["float_features"]):
        for b in f["borders"]:
            assert any(abs(b - fb) < 1e-9 for fb in borders[fi]), (
                f"model border {b} (feature {fi}) missing from frozen border set"
            )

    np.save(os.path.join(HERE, "borders.npy"), borders)
    np.save(os.path.join(HERE, "X.npy"), x)
    np.save(os.path.join(HERE, "y.npy"), y)
    np.save(os.path.join(HERE, "predictions.npy"), predictions.astype(np.float64))

    config = {
        "scenario": SCENARIO,
        "requirement": "FPP-03 / FPP-04",
        "seed": 0,
        "catboost_version": "1.2.10",
        "thread_count": 1,
        "n_rows": N_ROWS,
        "mean_y": float(y.mean()),
        "boost_from_average_delta": float(max_delta),
        "overflow_margin": (
            "n*max(w)*max(|der1|) <= 64*1.0*10 = 640 << 2^33 ~= 8.6e9 "
            "(margin > 1.3e7x) — the device fixed-point histogram precondition"
        ),
        "border_count_note": (
            "border_count=15 -> n_bins=16 -> pad_hist_line_bins(16)=32, an admitted "
            "resident histogram line width (PLAN blocker B-3, resolved)"
        ),
        "description": (
            "NON-ZERO-BIAS device-parity fixture, SymmetricTree arm. 64 rows, two float "
            "columns, border_count=15, RMSE depth-3 x3 iterations, Plain, no sampling, "
            "uniform weights, Gradient leaf, L2 score, boost_from_average=True so the "
            "starting approximant is mean(y) != 0. Pins the upstream fit that "
            "DeviceTrainConfig.bias (FPP-01) must reproduce at <=1e-5 once the "
            "device_host_eligible `bias == 0.0` clause (FPP-02) is removed. NEVER "
            "regenerated in CI."
        ),
        "params": PARAMS,
        "npy_schema": {
            "borders.npy": "[2,15] f64 — the FULL frozen quantization border set (feed to train)",
            "X.npy": "[N,2] f32 — float feature matrix",
            "y.npy": "[N] f64 — RMSE target, |y| < 10 and |mean(y)| > 0.5 by construction",
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
        f"wrote {SCENARIO}: {N_ROWS} rows, mean(y)={y.mean():.6f}, "
        f"bias-on-vs-off max|Δ|={max_delta:.6f}, predictions.std()={predictions.std():.6f}"
    )


if __name__ == "__main__":
    main()
