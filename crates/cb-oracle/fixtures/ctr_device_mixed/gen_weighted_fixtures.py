"""ADDITIVE weighted-composition artifacts for `ctr_device_mixed` (GDC-19.2 / T23).

FROZEN GENERATOR — adds `weights.npy`, `borders_weighted.npy` and
`predictions_weighted.npy` alongside the base fixture WITHOUT touching the
existing frozen artifacts (same X/X_cat/y, same params, plus a non-uniform
`sample_weight`). The T23 positive composition test (weighted × CTR admits to
the device TOGETHER) compares against these at <=1e-5. CI never runs this.

The weighted pool is quantized separately (weights can shift GreedyLogSum
borders), so the weighted run freezes its OWN full border set; the weighted fit
on the quantized pool is asserted bit-identical to the raw fit, exactly like the
base generator.

Overflow margin (SPEC §9): Logloss der in (-1, 1), n=64, max(w)=3.0 ⇒
n·max(w)·max|der| < 192 ≪ 2^33.
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

WEIGHT_CYCLE = [0.5, 1.0, 2.0, 3.0]


def main():
    with open(os.path.join(HERE, "config.json")) as fh:
        config = json.load(fh)
    params = dict(config["params"])

    x = np.load(os.path.join(HERE, "X.npy"))
    cat = np.load(os.path.join(HERE, "X_cat.npy"))
    y = np.load(os.path.join(HERE, "y.npy")).astype(np.int32)
    n = len(y)
    weights = np.array([WEIGHT_CYCLE[i % len(WEIGHT_CYCLE)] for i in range(n)], dtype=np.float64)

    def make_pool():
        df = [[int(cat[i]), float(x[i, 0]), float(x[i, 1])] for i in range(n)]
        return Pool(df, label=y, cat_features=[0], weight=weights)

    qpool = make_pool()
    qpool.quantize(border_count=params["border_count"])
    borders_tsv = os.path.join(HERE, "borders_weighted.tsv")
    qpool.save_quantization_borders(borders_tsv)
    per_feature = {}
    with open(borders_tsv) as fh:
        for line in fh:
            fi, bv = line.split()
            per_feature.setdefault(int(fi), []).append(float(bv))
    os.remove(borders_tsv)
    for fi in sorted(per_feature):
        assert len(per_feature[fi]) == params["border_count"], (
            f"feature {fi}: {len(per_feature[fi])} borders != {params['border_count']}"
        )
    borders = np.array(
        [sorted(per_feature[fi]) for fi in sorted(per_feature)], dtype=np.float64
    )

    model = CatBoost(params)
    model.fit(qpool)
    predictions = model.predict(make_pool(), prediction_type="RawFormulaVal")

    raw_model = CatBoost(params)
    raw_model.fit(make_pool())
    raw_preds = raw_model.predict(make_pool(), prediction_type="RawFormulaVal")
    assert np.abs(predictions - raw_preds).max() == 0.0, "border freezing unsound"

    # Guards: the weighted model still uses a CTR split AND differs from the
    # unweighted base predictions (else the composition test is vacuous).
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        mj_path = tf.name
    model.save_model(mj_path, format="json")
    with open(mj_path) as fh:
        mj = json.load(fh)
    os.unlink(mj_path)
    ctrs = mj["features_info"].get("ctrs", [])
    assert any(c["ctr_type"] == "Borders" for c in ctrs), "weighted model lost its CTR split"
    base = np.load(os.path.join(HERE, "predictions.npy"))
    assert np.abs(predictions - base).max() > 1e-6, "weighted == unweighted predictions"

    np.save(os.path.join(HERE, "weights.npy"), weights)
    np.save(os.path.join(HERE, "borders_weighted.npy"), borders)
    np.save(os.path.join(HERE, "predictions_weighted.npy"), predictions.astype(np.float64))

    out = subprocess.run(
        ["git", "status", "--porcelain", FIXTURES], capture_output=True, text=True, cwd=FIXTURES
    ).stdout
    offenders = [l for l in out.splitlines() if l.strip() and SCENARIO not in l]
    if offenders:
        print("corpus contamination:")
        for l in offenders:
            print("   ", l)
        sys.exit(1)
    print(
        f"wrote weighted composition artifacts: weighted-vs-base max|Δ|="
        f"{np.abs(predictions - base).max():.6f}"
    )


if __name__ == "__main__":
    main()
