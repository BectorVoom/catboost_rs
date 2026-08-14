#!/usr/bin/env python3
"""`feature_border_type` build cost — CPU, no GPU required.

`feature_border_type` chooses a HOST-side quantization algorithm, so its cost is paid the
same whether the fit later runs on the CPU or commits to a device. That makes it measurable
without a GPU box, and worth measuring separately from the GPU grid.

Three passes, because the headline number alone would be misleading:

1. `cost()` — every type against `GreedyLogSum` (the catboost default), everything else
   held fixed, so the DIFFERENCE between cells is the border-build cost. Reports
   median/min/max and refuses to call a type slower when its range overlaps the baseline's.

2. `attribution()` — is the extra time one-time PREP or per-iteration work? If it were in
   the grow loop it would scale with iterations. Border building is paid once, so the
   absolute delta must stay flat while the RATIO collapses. Without this pass an 8x figure
   from a short 10-iteration fit reads as "8x slower training", which it is not.

3. `vs_upstream()` — is the cost OURS or the algorithm's? The exact DP is inherently
   dearer than the greedy heap, so some penalty is expected and the raw number cannot
   settle it. Each engine is measured against its OWN `GreedyLogSum`, so absolute engine
   speed cancels and only the border-type penalty is compared.

Usage:
    python bench/param_wave_gpu_speed/border_cost_cpu.py [cost|attribution|vs-upstream|all]
"""

import sys
import time

import numpy as np

import catboost_rs

N, F, BORDERS, ITERS, REPEATS = 200_000, 20, 254, 10, 3

#: All seven, greedy-heap and exact-DP alike.
TYPES = ["Median", "GreedyLogSum", "UniformAndQuantiles", "MinEntropy",
         "MaxLogSum", "Uniform", "GreedyMinEntropy"]

#: `border_count = 254` is the upstream MAXIMUM on purpose: the DP's cost grows with the
#: border count, so this is the pessimistic end of the axis rather than a typical setting.
COMMON = dict(
    depth=6, learning_rate=0.03, l2_leaf_reg=3.0, border_count=BORDERS,
    random_seed=42, random_strength=0, bootstrap_type="No",
    boost_from_average=False, leaf_estimation_method="Gradient", score_function="L2",
)


def corpus():
    rng = np.random.RandomState(42)
    X = rng.randn(N, F).astype(np.float32)
    w = rng.randn(F).astype(np.float32)
    y = (X.dot(w) + rng.randn(N).astype(np.float32) * 0.1).astype(np.float32)
    return X, y


def _times(make, X, y, repeats=REPEATS):
    ts = []
    for _ in range(repeats):
        m = make()
        t0 = time.time()
        m.fit(X, y)
        ts.append(time.time() - t0)
    ts.sort()
    return ts


def cost():
    X, y = corpus()
    print(f"n={N} f={F} border_count={BORDERS} iterations={ITERS} repeats={REPEATS}\n")
    res = {}
    for t in TYPES:
        ts = _times(
            lambda t=t: catboost_rs.CatBoostRegressor(
                iterations=ITERS, feature_border_type=t, **COMMON),
            X, y,
        )
        res[t] = ts
        print(f"{t:<22} median {ts[len(ts)//2]:6.2f}s   min {ts[0]:6.2f}  max {ts[-1]:6.2f}")

    base = res["GreedyLogSum"]
    bm = base[len(base) // 2]
    print("\nrelative to GreedyLogSum (the catboost default):")
    for t in TYPES:
        ts = res[t]
        # Ranges that overlap the baseline's are NOT claimed as a difference.
        overlap = not (ts[0] > base[-1] or ts[-1] < base[0])
        tag = "  (ranges overlap — within noise)" if overlap and t != "GreedyLogSum" else ""
        print(f"  {t:<22} {ts[len(ts)//2]/bm:5.2f}x{tag}")


def attribution():
    X, y = corpus()
    print(f"n={N} f={F} border_count={BORDERS}, median of {REPEATS}\n")
    print(f"{'iters':>6} {'GreedyLogSum':>13} {'MinEntropy':>12} {'delta':>8} {'ratio':>7}")
    for iters in (10, 40, 160):
        g = _times(lambda: catboost_rs.CatBoostRegressor(
            iterations=iters, feature_border_type="GreedyLogSum", **COMMON), X, y)
        e = _times(lambda: catboost_rs.CatBoostRegressor(
            iterations=iters, feature_border_type="MinEntropy", **COMMON), X, y)
        gm, em = g[len(g) // 2], e[len(e) // 2]
        print(f"{iters:>6} {gm:>12.2f}s {em:>11.2f}s {em-gm:>7.2f}s {em/gm:>6.2f}x")
    print("\nA FLAT delta with a collapsing ratio means one-time border PREP.")
    print("A growing delta would mean the cost is in the grow loop instead.")


def vs_upstream():
    from catboost import CatBoostRegressor as OfficialReg

    X, y = corpus()
    subset = ["GreedyLogSum", "MinEntropy", "MaxLogSum"]
    rs, off = {}, {}
    for t in subset:
        r = _times(lambda t=t: catboost_rs.CatBoostRegressor(
            iterations=ITERS, feature_border_type=t, **COMMON), X, y)
        o = _times(lambda t=t: OfficialReg(
            iterations=ITERS, feature_border_type=t, verbose=False, **COMMON), X, y)
        rs[t], off[t] = r[len(r) // 2], o[len(o) // 2]

    print(f"n={N} f={F} border_count={BORDERS} iterations={ITERS}, median of {REPEATS}\n")
    print(f"{'border type':<16} {'cb-rs':>9} {'official':>10} {'cb-rs pen':>11} {'off pen':>9}")
    for t in subset:
        print(f"{t:<16} {rs[t]:>8.2f}s {off[t]:>9.2f}s "
              f"{rs[t]/rs['GreedyLogSum']:>10.2f}x {off[t]/off['GreedyLogSum']:>8.2f}x")
    print("\n'pen' is each engine's cost RELATIVE TO ITS OWN GreedyLogSum, so absolute")
    print("engine speed cancels and only the border-type penalty is compared.")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("cost", "all"):
        cost()
        print()
    if which in ("attribution", "all"):
        attribution()
        print()
    if which in ("vs-upstream", "all"):
        vs_upstream()
