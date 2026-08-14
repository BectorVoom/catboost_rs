#!/usr/bin/env python3
"""Preflight BOTH arms of the param-wave grid on local hardware, before spending a GPU box.

Every cell's kwargs must be ACCEPTED by the surface that will receive them, and — on the
catboost-rs side — must actually MOVE the predictions. This is not ceremony:

  * A kwarg typo or an unimplemented parameter is discovered here in seconds, instead of
    after the ~25-minute build on the GPU box.
  * It is what caught `leaf_estimation_iterations` being implemented in the engine and on
    the Rust builder, but absent from the Python surface's IMPLEMENTED list — so `fit`
    rejected it as a "parity gap" for a parameter that was in fact implemented. No
    Rust-side oracle could see that seam, because none of them cross the Python surface.
  * The discrimination check (predictions must differ from the baseline cell) catches the
    quieter failure: a parameter ACCEPTED and then dropped on the floor, which a
    does-fit-return test cannot distinguish from a working one.

The official arm is checked on CPU. A kwarg official CatBoost rejects on CPU it will also
reject on GPU; the converse does not hold, so a GPU-only rejection is left for the harness
to discover and report with its real message rather than guessed at here.

Usage:
    python bench/param_wave_gpu_speed/preflight.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402

rng = np.random.default_rng(0)
X = rng.normal(size=(600, 8)).astype(np.float32)
y = (X[:, 0] * 2 - X[:, 1]).astype(np.float32)
X_nan = X.copy()
X_nan[rng.random(600) < 0.10, 0] = np.nan


def preflight_rs():
    import catboost_rs

    failures, preds = [], {}
    for cell in bench.build_grid():
        kw = dict(cell["kwargs"], iterations=3, border_count=16)
        data = X_nan if cell["nan_pool"] else X
        try:
            m = catboost_rs.CatBoostRegressor(**kw)
            m.fit(data, y)
            p = np.asarray(m.predict(data)).ravel()
            preds[cell["name"]] = p
            print(f"  {cell['name']:<40} OK")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{cell['name']}: {type(exc).__name__}: {exc}")
            print(f"  {cell['name']:<40} FAIL {type(exc).__name__}: {str(exc)[:160]}")

    # Discrimination: a parameter that is accepted but dropped produces predictions
    # identical to the baseline. `nan_mode=Min` is EXCLUDED — Min is the upstream default,
    # so it SHOULD reproduce the baseline recipe on its own pool, and demanding otherwise
    # would be asserting a bug. `GreedyMinEntropy` is excluded for the same class of
    # reason: it coincides with `GreedyLogSum` whenever the penalty does not separate,
    # which on a small smooth corpus it does not (the border oracle uses a purpose-built
    # duplicate-run corpus to tell them apart).
    base = preds.get("baseline")
    inert_by_design = {"baseline", "nan_mode=Min", "feature_border_type=GreedyLogSum",
                       "feature_border_type=GreedyMinEntropy"}
    if base is not None:
        for name, p in preds.items():
            if name in inert_by_design or len(p) != len(base):
                continue
            if np.allclose(p, base):
                failures.append(f"{name}: accepted but did NOT change the model")
                print(f"  {name:<40} INERT — accepted but changed nothing")
    return failures


def preflight_official():
    from catboost import CatBoostRegressor

    failures = []
    for cell in bench.build_grid():
        if cell.get("official_na"):
            print(f"  {cell['name']:<40} SKIP (declared N/A)")
            continue
        kw = dict(cell["kwargs"], iterations=2, border_count=16, verbose=False)
        try:
            CatBoostRegressor(**kw).fit(X, y)
            print(f"  {cell['name']:<40} OK")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{cell['name']}: {type(exc).__name__}: {exc}")
            print(f"  {cell['name']:<40} REJECTED {type(exc).__name__}: {str(exc)[:160]}")
    return failures


if __name__ == "__main__":
    print("catboost-rs arm:")
    rs = preflight_rs()
    print("\nofficial CatBoost arm (CPU):")
    off = preflight_official()
    print()
    if rs or off:
        print(f"{len(rs) + len(off)} PROBLEMS — do not spend a GPU session yet:")
        for f in rs + off:
            print(f"  - {f}")
        sys.exit(1)
    print("both arms accept every cell, and every non-inert cell moves the model")
