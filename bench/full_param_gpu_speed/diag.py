#!/usr/bin/env python3
# SPD-03 targeted diagnostic: the 1M x 50 RMSE / SymmetricTree / depth-6 / 30-iteration
# cell (the grid's worst ratio) profiled with CB_GPU_PROF=1 in FRESH subprocesses, so the
# per-stage attribution lines (begin-inner / qpack-fill / tree / tree-host) land in a raw
# log the FINDINGS can cite. Assumes `catboost_rs` is already installed (run AFTER
# bench.py, which builds + installs the wheel).
#
# Two runs:
#   * diag_cold.txt   — XDG_CACHE_HOME points at a fresh scratch dir, so CubeCL's disk
#                       compilation cache is EMPTY: this is the true first-fit-on-a-machine
#                       cost, and what the fit-entry background warm-up must hide.
#   * diag_repeat.txt — default environment: the disk cache was populated by the grid
#                       process (and/or the cold run's scratch is not shared), so this is
#                       the every-later-process cost real users see.
#
# Do-not-fabricate: the script only reports what the subprocesses print; a failed run
# leaves its log with the failure output.

import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("CB_BENCH_REPO", os.path.abspath(os.path.join(HERE, "..", "..")))
WORK = os.environ.get("CB_BENCH_WORK", "/kaggle/working")
CELL = "SymmetricTree|RMSE|unw|noctr|1000k"


def main():
    sys.path.insert(0, HERE)
    import bench  # noqa: E402 — the committed grid definition IS the recipe source

    cells = {c["name"]: c for c in bench.build_grid()}
    cell = cells.get(CELL)
    if cell is None:
        print(f"diag: cell {CELL!r} not in grid — nothing to profile", flush=True)
        return 1

    probe = os.path.join(WORK, "_diag_fit.py")
    with open(probe, "w") as fh:
        fh.write(
            "import os, sys, json, time, numpy as np, catboost_rs\n"
            f"kw = json.loads({json.dumps(json.dumps(cell['kwargs']))})\n"
            f"n_rows, n_features = {cell['shape']}\n"
            f"sys.path.insert(0, {json.dumps(os.path.join(REPO, 'bench'))})\n"
            "import generator as gen\n"
            f"X, y = gen.generate(n_rows, n_features, seed={bench.RANDOM_SEED})\n"
            "m = catboost_rs.CatBoostRegressor(**kw)\n"
            "t0 = time.time()\n"
            "m.fit(X, y)\n"
            "print(f'WALLCLOCK_FIT_SECONDS={time.time() - t0:.4f}', flush=True)\n"
            "_ = m.predict(X[:1024])\n"
        )

    diag_dir = os.path.join(WORK, "diag")
    os.makedirs(diag_dir, exist_ok=True)
    for label, extra_env in (
        ("cold", {"XDG_CACHE_HOME": tempfile.mkdtemp(prefix="cb_diag_coldcache_")}),
        ("repeat", {}),
    ):
        env = dict(os.environ, CB_GPU_PROF="1", **extra_env)
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, probe], env=env, capture_output=True, text=True, timeout=3600,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        path = os.path.join(diag_dir, f"diag_{label}.txt")
        with open(path, "w") as fh:
            fh.write(f"# diag run={label} rc={proc.returncode} "
                     f"elapsed={time.time() - t0:.1f}s cell={CELL}\n")
            fh.write(out)
        print(f"diag[{label}] rc={proc.returncode} -> {path}", flush=True)
        print(out[-2000:], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
