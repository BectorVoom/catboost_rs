#!/usr/bin/env python3
"""What does the device-decline cliff COST? — self-relative, so it needs no official arm.

`model_shrink_rate != 0` and `leaf_estimation_iterations > 1` both route the fit to the
CPU grower (`device_host_eligible`, `crates/cb-train/src/boosting.rs`). Both declines are
correct — the alternative is a wrong model — and both are asserted in
`crates/cb-train/tests/string_param_device_routing_test.rs`. What is documented nowhere a
user would look is what setting one COSTS on a GPU box.

That cost is a comparison of catboost-rs against ITSELF (device baseline vs the same fit
pushed onto the CPU grower), so unlike the border-cost work it needs no official CatBoost
arm — and therefore runs on **any** backend, ROCm included, where official CatBoost has no
GPU trainer to compare against at all.

Device activation is OBSERVED, not assumed: each cell is re-run as a subprocess under
`CB_GPU_PROF=1` and the `CB_GPU_PROF tree` lines are counted. A baseline that shows no
device lines makes every ratio below meaningless, and this script says so and exits rather
than reporting a number.

Usage:
    # build a backend-enabled wheel first, e.g.
    #   maturin build --release --no-default-features --features rocm \
    #       -m crates/catboost-rs-py/Cargo.toml
    python bench/param_wave_gpu_speed/decline_cliff.py
"""

import json
import os
import subprocess
import sys
import time

import numpy as np

N, F, ITERS, DEPTH, BORDERS = 300_000, 50, 30, 6, 128
SEED = 42

#: The reported interval is deliberately CONSERVATIVE — `min(decline)/max(baseline)` to
#: `max(decline)/min(baseline)` — so a single slow repeat on either side widens it. At 3
#: repeats that interval spans 1.0 even when the medians differ by 35%, which correctly
#: refuses the claim but does not settle it. Raise `CB_BENCH_REPEATS` to tighten.
REPEATS = int(os.environ.get("CB_BENCH_REPEATS", "3"))

BASE = dict(
    iterations=ITERS, depth=DEPTH, learning_rate=0.03, l2_leaf_reg=3.0,
    border_count=BORDERS, random_seed=SEED, random_strength=0,
    bootstrap_type="No", boost_from_average=False,
    leaf_estimation_method="Gradient", score_function="L2",
)

CELLS = [
    ("baseline", {}, True,
     "the device-eligible reference every decline row is measured against"),
    ("model_shrink_rate=0.2", {"model_shrink_rate": 0.2}, False,
     "the shrink rescales the running approx each iteration, but the device keeps its "
     "approx RESIDENT and never reads the host copy back per tree"),
    ("leaf_estimation_iterations=3",
     {"leaf_estimation_iterations": 3, "leaf_estimation_backtracking": "No"}, False,
     "the accumulate-and-recompute loop lives in the CPU leaf-value section, which the "
     "device branch skips entirely"),
]


def corpus():
    rng = np.random.RandomState(SEED)
    X = rng.randn(N, F).astype(np.float32)
    w = rng.randn(F).astype(np.float32)
    y = (X.dot(w) + rng.randn(N).astype(np.float32) * 0.1).astype(np.float32)
    return X, y


def device_probe(kwargs):
    """Count CB_GPU_PROF tree lines over a short fit, in a subprocess."""
    src = (
        "import json, numpy as np, catboost_rs\n"
        f"kw = json.loads({json.dumps(json.dumps(dict(BASE, **kwargs)))})\n"
        "kw['iterations'] = 2\n"
        f"rng = np.random.RandomState({SEED})\n"
        f"X = rng.randn({N}, {F}).astype(np.float32)\n"
        f"w = rng.randn({F}).astype(np.float32)\n"
        f"y = (X.dot(w) + rng.randn({N}).astype(np.float32) * 0.1).astype(np.float32)\n"
        "catboost_rs.CatBoostRegressor(**kw).fit(X, y)\n"
    )
    p = subprocess.run([sys.executable, "-c", src],
                       env=dict(os.environ, CB_GPU_PROF="1"),
                       capture_output=True, text=True, timeout=3600)
    return (p.stdout + p.stderr).count("CB_GPU_PROF tree")


def main():
    import catboost_rs

    X, y = corpus()
    print(f"n={N} f={F} iterations={ITERS} depth={DEPTH} border_count={BORDERS}, "
          f"median of {REPEATS}\n")

    rows = []
    for name, extra, expect_device, why in CELLS:
        kwargs = dict(BASE, **extra)
        lines = device_probe(extra)
        observed = lines > 0
        ts = []
        for _ in range(REPEATS):
            m = catboost_rs.CatBoostRegressor(**kwargs)
            t0 = time.time()
            m.fit(X, y)
            ts.append(time.time() - t0)
        ts.sort()
        rows.append(dict(name=name, median=ts[len(ts) // 2], lo=ts[0], hi=ts[-1],
                         device=observed, lines=lines, expect=expect_device, why=why))
        print(f"{name:<32} {ts[len(ts)//2]:6.2f}s  device={'yes' if observed else 'NO '} "
              f"({lines} tree lines, expected {'yes' if expect_device else 'NO'})")

    bad = [r for r in rows if r["device"] != r["expect"]]
    if bad:
        print("\nHARNESS FAILURE — routing does not match what the tests assert:")
        for r in bad:
            print(f"  {r['name']}: expected device={r['expect']}, observed {r['device']}")
        print("Every ratio below would be uninterpretable; refusing to report one.")
        return 1

    base = rows[0]
    if not base["device"]:
        print("\nBASELINE DID NOT REACH THE DEVICE — no decline cost can be quoted.")
        return 1

    print("\ncost of the decline (vs the device baseline):")
    for r in rows[1:]:
        # A ratio whose range spans 1.0 is within noise and is not claimed.
        lo, hi = r["lo"] / base["hi"], r["hi"] / base["lo"]
        tag = "  (within noise)" if lo < 1.0 < hi else ""
        print(f"  {r['name']:<32} {r['median']/base['median']:5.2f}x  "
              f"[{lo:.2f}, {hi:.2f}]{tag}")
        print(f"      {r['why']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
