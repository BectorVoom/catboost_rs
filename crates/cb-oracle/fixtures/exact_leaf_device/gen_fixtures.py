"""Generate the EXACT-leaf device-parity fixtures (FPP-07/T02): `mae/` and `quantile07/`.

FROZEN GENERATOR — the committed artifacts under `mae/` and `quantile07/` are the
GROUND TRUTH the device Exact order-statistic leaf (FPP-05/FPP-06,
`DeviceTrainConfig.exact_leaf` + `quantile_alpha`/`quantile_delta`) is compared
against at <=1e-5. CI does NOT run this script (no `catboost` install in CI) and
never regenerates these fixtures.

# Why NEW fixtures rather than reusing `quantile_alpha05_mae/` + `quantile_alpha07/`

Those two already pin `leaf_estimation_method="Exact"` with an otherwise
device-shaped recipe, but (PLAN V-7, verified):

  - they ship no `predictions.npy` / `X.npy` / `y.npy` — nothing to compare against;
  - they pin no `border_count`, inheriting upstream's default 254 (255 bins), which
    `pad_hist_line_bins` rejects for the resident histogram line widths;
  - their only consumer imports `CpuBackend`, so under `--no-default-features
    --features rocm` it fails E0432 and cannot run at all.

Reuse is not possible. These fixtures are device-shaped from the start.

# The admissible `border_count` (PLAN blocker B-3, RESOLVED)

`pad_hist_line_bins` (`cb-backend/src/gpu_runtime/session.rs`) rounds a quantized bin
count UP to the next dispatched resident line width `1 << bits`, bits 5..=8 →
{32, 64, 128, 256}, and returns `None` only for `n_bins > 256`. `border_count=15` ⇒
`n_bins = 16` ⇒ `pad_hist_line_bins(16) = Some(32)`, an admitted width; the padding
cells stay zero and their phantom borders are excluded from the split argmin, so
padding is score-invariant. 15 is therefore admissible for this 2-float pool, and is
also what every currently-green device fixture uses.

# Why the pair, and why the alpha differs

A device path that computed an Exact leaf but silently ignored `quantile_alpha`
would pass a single-fixture test. `mae/` (α = 0.5 by MAE's definition) and
`quantile07/` (α = 0.7) predict differently, and the Rust smoke test asserts
`max|Δ| > 1e-6` between them — so α is provably load-bearing.

# Fixed-point overflow margin (SPEC §9, mandatory)

The device fixed-point histogram requires |Σ| < 2^33 ≈ 8.6e9 (kernels.rs). Weights
are UNIFORM 1.0 here, and MAE/Quantile der1 ∈ {-1, +1} (or {α-1, α}), so
`n · max(w) · max(|der1|) <= 64 · 1.0 · 1.0 = 64` — a margin > 1.3e8x under the
bound, the widest of any fixture in this phase.

# Pinned recipe (identical to `bias_device_sym/` except where noted)

  - `catboost==1.2.10`, `numpy.random.RandomState(0)`, `thread_count=1`.
  - 64 rows, TWO float columns in [0, 1), ZERO categorical columns.
  - `border_count=15` (16 bins), iterations=3, depth=3, lr=0.3, l2=3.0.
  - Plain / bootstrap No / random_strength 0 / L2 score / SymmetricTree.
  - `boost_from_average=False` — isolate the exact-leaf axis from Track A (bias).
  - `leaf_estimation_method="Exact"`, `leaf_estimation_iterations=1`.
  - `mae/`:        `loss_function="MAE"`.
  - `quantile07/`: `loss_function="Quantile:alpha=0.7"`.
"""

import json
import os
import subprocess
import sys

import numpy as np
from catboost import CatBoost, Pool

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.dirname(HERE)
SCENARIO = "exact_leaf_device"

N_ROWS = 64
N_FLOAT = 2

BASE_PARAMS = {
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
    "leaf_estimation_method": "Exact",
    "leaf_estimation_iterations": 1,
    "score_function": "L2",
    "grow_policy": "SymmetricTree",
    "random_seed": 0,
    "thread_count": 1,
    "verbose": False,
}

ARMS = {
    "mae": {"loss_function": "MAE"},
    "quantile07": {"loss_function": "Quantile:alpha=0.7"},
}


def make_data():
    rng = np.random.RandomState(0)
    x = rng.rand(N_ROWS, N_FLOAT).astype(np.float32)
    # Bounded target: |y| <= 3 + 2 + 0.5 + noise(~2) < 10 by construction.
    y = (
        3.0 * x[:, 0]
        - 2.0 * x[:, 1]
        + 0.5 * np.sin(6.0 * x[:, 0])
        + 0.5 * rng.randn(N_ROWS)
    ).astype(np.float64)
    assert np.abs(y).max() < 10.0, "target bound backs the 2^33 margin arithmetic"
    return x, y


def freeze_borders(x, y, border_count, out_dir):
    """Quantize ONCE and freeze the FULL border set (model.json keeps only the pruned
    USED subset, which is too few for the device `n_bins` arithmetic)."""
    pool = Pool(x, label=y)
    pool.quantize(border_count=border_count)
    borders_tsv = os.path.join(out_dir, "borders.tsv")
    pool.save_quantization_borders(borders_tsv)
    per_feature = {}
    with open(borders_tsv) as fh:
        for line in fh:
            fi, bv = line.split()
            per_feature.setdefault(int(fi), []).append(float(bv))
    os.remove(borders_tsv)
    assert sorted(per_feature) == list(range(N_FLOAT))
    for fi, bs in per_feature.items():
        assert len(bs) == border_count, (
            f"feature {fi}: expected {border_count} borders, got {len(bs)}"
        )
    borders = np.array([sorted(per_feature[fi]) for fi in range(N_FLOAT)], dtype=np.float64)
    return pool, borders


def build_arm(arm, overrides, x, y):
    out_dir = os.path.join(HERE, arm)
    os.makedirs(out_dir, exist_ok=True)
    params = dict(BASE_PARAMS, **overrides)

    pool, borders = freeze_borders(x, y, params["border_count"], out_dir)

    model = CatBoost(params)
    model.fit(pool)
    eval_pool = Pool(x, label=y)
    predictions = model.predict(eval_pool, prediction_type="RawFormulaVal")

    # --- anti-false-pass guard: Exact must differ from the Gradient default -----
    gradient = CatBoost(dict(params, leaf_estimation_method="Gradient"))
    gradient.fit(Pool(x, label=y))
    gradient_preds = gradient.predict(eval_pool, prediction_type="RawFormulaVal")
    exact_delta = float(np.abs(predictions - gradient_preds).max())
    assert exact_delta > 1e-6, (
        f"{arm}: Exact and Gradient leaves agree (max|Δ|={exact_delta:.3e}) — the "
        "fixture cannot discriminate the exact order-statistic leaf"
    )
    assert predictions.std() > 1e-6, f"{arm}: degenerate constant predictions"

    model_json_path = os.path.join(out_dir, "model.json")
    model.save_model(model_json_path, format="json")
    with open(model_json_path) as fh:
        model_json = json.load(fh)
    for fi, f in enumerate(model_json["features_info"]["float_features"]):
        for b in f["borders"]:
            assert any(abs(b - fb) < 1e-9 for fb in borders[fi]), (
                f"{arm}: model border {b} (feature {fi}) missing from frozen border set"
            )

    np.save(os.path.join(out_dir, "borders.npy"), borders)
    np.save(os.path.join(out_dir, "X.npy"), x)
    np.save(os.path.join(out_dir, "y.npy"), y)
    np.save(os.path.join(out_dir, "predictions.npy"), predictions.astype(np.float64))

    config = {
        "scenario": f"{SCENARIO}/{arm}",
        "requirement": "FPP-07 / FPP-08",
        "seed": 0,
        "catboost_version": "1.2.10",
        "thread_count": 1,
        "n_rows": N_ROWS,
        "exact_vs_gradient_max_delta": exact_delta,
        "overflow_margin": (
            "n*max(w)*max(|der1|) <= 64*1.0*1.0 = 64 << 2^33 ~= 8.6e9 (margin > 1.3e8x) "
            "— MAE/Quantile der1 is bounded by 1 in magnitude"
        ),
        "border_count_note": (
            "border_count=15 -> n_bins=16 -> pad_hist_line_bins(16)=32, an admitted "
            "resident histogram line width (PLAN blocker B-3, resolved)"
        ),
        "description": (
            f"EXACT-leaf device-parity fixture, {params['loss_function']} arm. 64 rows, "
            "two float columns, border_count=15, depth-3 x3 iterations, Plain, no "
            "sampling, uniform weights, bias 0, L2 score, SymmetricTree, "
            "leaf_estimation_method=Exact. Pins the upstream order-statistic leaf the "
            "device exact_leaf path (FPP-05) must reproduce at <=1e-5. Paired with the "
            "sibling arm so that quantile_alpha is provably load-bearing. NEVER "
            "regenerated in CI."
        ),
        "params": params,
        "npy_schema": {
            "borders.npy": "[2,15] f64 — the FULL frozen quantization border set (feed to train)",
            "X.npy": "[N,2] f32 — float feature matrix",
            "y.npy": "[N] f64 — target, |y| < 10 by construction",
            "predictions.npy": "[N] f64 — RawFormulaVal (<=1e-5 gate)",
        },
        "note": (
            "FROZEN. Generated once by this directory's parent gen_fixtures.py and "
            "NEVER regenerated in CI. Regenerating invalidates the <=1e-5 gate."
        ),
    }
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)
        fh.write("\n")

    return predictions, exact_delta


def main():
    x, y = make_data()
    preds = {}
    for arm, overrides in ARMS.items():
        preds[arm], delta = build_arm(arm, overrides, x, y)
        print(f"  {arm}: exact-vs-gradient max|Δ|={delta:.6f}, std={preds[arm].std():.6f}")

    alpha_delta = float(np.abs(preds["mae"] - preds["quantile07"]).max())
    assert alpha_delta > 1e-6, (
        f"MAE and Quantile:alpha=0.7 predictions agree (max|Δ|={alpha_delta:.3e}) — a "
        "device path ignoring quantile_alpha would pass; the pair is vacuous"
    )

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

    print(f"wrote {SCENARIO}/{{mae,quantile07}}: alpha-discrimination max|Δ|={alpha_delta:.6f}")


if __name__ == "__main__":
    main()
