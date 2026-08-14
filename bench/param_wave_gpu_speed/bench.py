#!/usr/bin/env python3
"""GPU speed measurement for the parameter-wireup wave's perf-significant params.

This wave added `feature_border_type`, `nan_mode`, `model_shrink_rate`/`model_shrink_mode`,
`leaf_estimation_iterations`, `leaf_estimation_backtracking`, `random_score_type`, the two
CTR-mode params and the logging family. Most of those cost nothing measurable. TWO groups
do, and they are the only reason this harness exists:

A. **The device-decline cliff.** `model_shrink_rate != 0` and `leaf_estimation_iterations >
   1` DECLINE to the CPU grower (`device_host_eligible` in `crates/cb-train/src/boosting.rs`;
   the declines are asserted in `crates/cb-train/tests/string_param_device_routing_test.rs`).
   Both are ordinary-looking knobs a user would set without expecting a path change, so the
   cost of setting one on a GPU fit is the single most actionable number this wave can
   produce. It is measured SELF-RELATIVE — catboost-rs device baseline vs catboost-rs with
   the declining param — which needs no official arm to be meaningful.

B. **Border-build cost.** `feature_border_type` picks among a greedy heap
   (`GreedyLogSum`/`GreedyMinEntropy`), an exact O(values x borders) dynamic program
   (`MaxLogSum`/`MinEntropy`), and three cheap analytic rules (`Median`, `Uniform`,
   `UniformAndQuantiles`). Border building is HOST prep, and host prep is a large share of a
   GPU fit's wall clock at these shapes (see bench/RESULTS.md), so an exact-DP border type
   can cost real time without touching a kernel.

Disciplines inherited from `bench/full_param_gpu_speed/bench.py`, unchanged:

1. Device activation is OBSERVED per cell via `CB_GPU_PROF` tree lines, never assumed. A
   cell expected to commit that shows zero lines is a HARNESS FAILURE, not a slow row, and
   is reported as such.
2. Cells expected to DECLINE are equally observed: a decline cell that shows tree lines
   means `device_host_eligible` did not do what the routing tests say, and the run says so.
3. Both sides get the same explicit recipe; a recipe official CatBoost GPU cannot express
   is recorded `N/A` with the reason, never swapped for a different one.
4. Spread before headline: median/min/max over repeats, and a ratio whose spread crosses
   1.0 is reported *within noise* rather than as a win or a loss.
5. No number is invented. A failed build or a failed cell produces an error row.

Usage:
    python bench/param_wave_gpu_speed/bench.py --dry-run   # grid review, no GPU needed
    python bench/param_wave_gpu_speed/bench.py             # on a CUDA box (Colab)
"""

import json
import os
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORK = os.environ.get("CB_BENCH_OUT", "/content/bench_out")

# ── Pinned recipe ────────────────────────────────────────────────────────────────────
ITERS = 30
DEPTH = 6
LEARNING_RATE = 0.03
L2_LEAF_REG = 3.0
BORDER_COUNT = 128  # deliberately generous: border COST is one of the two things measured
RANDOM_SEED = 42
REPEATS = 3

#: At or above the D-10-09 device/CPU crossover (n = 100_000, bench/RESULTS.md). Below it
#: the device cannot win — launch-overhead physics, not a tuning gap.
SHAPE = (300_000, 50)

#: Wall-clock ceiling, leaving headroom inside a Colab session.
BUDGET_S = float(os.environ.get("CB_BENCH_BUDGET_S", 3 * 3600))

#: Every `feature_border_type` catboost 1.2.10 accepts.
BORDER_TYPES = (
    "Median",
    "GreedyLogSum",
    "UniformAndQuantiles",
    "MinEntropy",
    "MaxLogSum",
    "Uniform",
    "GreedyMinEntropy",
)

#: `feature_border_type` values official CatBoost REJECTS under `task_type='GPU'`.
#:
#: DELIBERATELY EMPTY. An earlier draft declared `GreedyMinEntropy` unsupported, but that
#: could not be checked from the development box (a ROCm rig with no CUDA driver, where
#: every `task_type='GPU'` call dies with "CUDA driver version is insufficient" regardless
#: of the parameter) — so the claim would have been an assumption dressed as data, and a
#: wrongly-declared N/A silently DROPS an official arm that would otherwise have run.
#:
#: Rejections are therefore DISCOVERED on the GPU box: the harness catches the exception,
#: records it in `official_error` with the message, and the report prints it in place of a
#: number. An arm that is genuinely unsupported shows up as its real error; one that is
#: supported gets measured.
GPU_UNSUPPORTED_BORDER_TYPES = ()


def _base_kwargs():
    return dict(
        iterations=ITERS,
        depth=DEPTH,
        learning_rate=LEARNING_RATE,
        l2_leaf_reg=L2_LEAF_REG,
        border_count=BORDER_COUNT,
        random_seed=RANDOM_SEED,
        random_strength=0,
        bootstrap_type="No",
        boost_from_average=False,
        leaf_estimation_method="Gradient",
        score_function="L2",
        loss_function="RMSE",
        grow_policy="SymmetricTree",
    )


def build_grid():
    """The grid, as a table. `expect_device` is the ASSERTION each cell carries."""
    cells = []

    # ── Group A: the device-decline cliff ────────────────────────────────────────────
    base = _base_kwargs()
    cells.append(
        dict(
            name="baseline",
            group="decline-cliff",
            kwargs=base,
            expect_device=True,
            nan_pool=False,
            note="the device-eligible reference every decline row is measured against",
        )
    )

    shrink = dict(_base_kwargs(), model_shrink_rate=0.2)
    cells.append(
        dict(
            name="model_shrink_rate=0.2",
            group="decline-cliff",
            kwargs=shrink,
            expect_device=False,
            nan_pool=False,
            note=(
                "declines: the shrink rescales the running approx each iteration, but the "
                "device keeps its approx RESIDENT and never reads the host copy back per "
                "tree"
            ),
        )
    )

    multistep = dict(
        _base_kwargs(),
        leaf_estimation_iterations=3,
        leaf_estimation_backtracking="No",
    )
    cells.append(
        dict(
            name="leaf_estimation_iterations=3",
            group="decline-cliff",
            kwargs=multistep,
            expect_device=False,
            nan_pool=False,
            note=(
                "declines: the accumulate-and-recompute loop lives in the CPU leaf-value "
                "section, which the device branch skips. `backtracking=No` is required — "
                "the default AnyImprovement is rejected outright at N>1"
            ),
        )
    )

    # ── Group B: border-build cost ───────────────────────────────────────────────────
    for bt in BORDER_TYPES:
        cells.append(
            dict(
                name=f"feature_border_type={bt}",
                group="border-build",
                kwargs=dict(_base_kwargs(), feature_border_type=bt),
                expect_device=True,
                nan_pool=False,
                official_na=(
                    "official CatBoost rejects this feature_border_type under "
                    "task_type='GPU'"
                    if bt in GPU_UNSUPPORTED_BORDER_TYPES
                    else None
                ),
                note="border building is HOST prep; exact-DP types can cost real wall clock",
            )
        )

    # ── Group C: nan_mode control ────────────────────────────────────────────────────
    # Expected to be indistinguishable from baseline. Measured anyway BECAUSE it is
    # expected to be free: `nan_mode=Max` is the one wave param that changes a per-object
    # inner loop (the quantizer's sentinel branch, on both the host and the QPACK-01 device
    # kernel), and "we assumed it was free" is exactly how a per-object regression ships.
    for mode in ("Min", "Max"):
        cells.append(
            dict(
                name=f"nan_mode={mode}",
                group="nan-mode",
                kwargs=dict(_base_kwargs(), nan_mode=mode),
                expect_device=True,
                nan_pool=True,
                note="NaN-bearing pool; Max appends the f32::MAX sentinel border",
            )
        )

    return cells


def projected_seconds(cells):
    """Rough dry-run projection. Decline cells run on CPU and are budgeted heavier."""
    n_rows = SHAPE[0]
    per_arm_device = 4.5 * (n_rows / 1e6)
    total = 0.0
    for c in cells:
        rs = per_arm_device * (6.0 if not c["expect_device"] else 1.0)
        official = 0.0 if c.get("official_na") else per_arm_device
        total += REPEATS * (rs + official) + per_arm_device  # + the CB_GPU_PROF probe
    return total


def dry_run():
    cells = build_grid()
    proj = projected_seconds(cells)
    print(f"catboost-rs param-wave GPU speed grid — {len(cells)} cells, shape {SHAPE}")
    print(f"projected {proj/60:.1f} min against a {BUDGET_S/60:.0f} min ceiling")
    print()
    for c in cells:
        na = c.get("official_na")
        print(
            f"  {c['name']:<34} group={c['group']:<14} "
            f"expect_device={str(c['expect_device']):<5} "
            f"nan_pool={str(c['nan_pool']):<5} "
            f"official={'N/A' if na else 'yes'}"
        )
    print()
    print("Assertions this grid carries (a violation is a HARNESS FAILURE, not a slow row):")
    print("  * every expect_device=True cell must show CB_GPU_PROF tree lines")
    print("  * every expect_device=False cell must show ZERO CB_GPU_PROF tree lines")
    if proj > BUDGET_S:
        print()
        print(f"WARNING: projection {proj/60:.1f} min EXCEEDS the {BUDGET_S/60:.0f} min ceiling")
        return 1
    return 0


def _sh(cmd, env=None, cwd=None, timeout=3600):
    if isinstance(cmd, str):
        cmd = ["bash", "-lc", cmd]
    p = subprocess.run(
        cmd, env=env, cwd=cwd, timeout=timeout,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    return p.returncode, p.stdout


def main():
    import numpy as np

    os.makedirs(WORK, exist_ok=True)
    result = {"provenance": {}, "grid": [], "errors": {}, "budget_s": BUDGET_S}
    started = time.time()

    def log(msg):
        print(msg, flush=True)

    rc, out = _sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    result["provenance"]["gpu"] = out.strip() if rc == 0 else f"nvidia-smi failed: {out[:200]}"
    log(f"GPU: {result['provenance']['gpu']}")

    rc, out = _sh("git rev-parse HEAD", cwd=REPO)
    result["provenance"]["commit"] = out.strip() if rc == 0 else "unknown"

    _sh("pip install -q maturin", timeout=1800)
    rc, out = _sh(
        ["maturin", "build", "--release", "--no-default-features", "--features", "cuda",
         "-m", os.path.join(REPO, "crates/catboost-rs-py/Cargo.toml")],
        cwd=REPO, timeout=5400,
    )
    result["build_ok"] = rc == 0
    result["build_tail"] = out[-4000:]
    if not result["build_ok"]:
        json.dump(result, open(os.path.join(WORK, "result.json"), "w"), indent=2)
        log("BUILD FAILED — no speed number is reported (do-not-fabricate).")
        return 1

    wheels = []
    for root, _dirs, files in os.walk(os.path.join(REPO, "target", "wheels")):
        wheels += [os.path.join(root, f) for f in files if f.endswith(".whl")]
    if not wheels:
        result["errors"]["wheel"] = "no wheel produced"
        json.dump(result, open(os.path.join(WORK, "result.json"), "w"), indent=2)
        return 1
    _sh(f"pip install -q --force-reinstall {sorted(wheels)[-1]}", timeout=1800)

    # Colab images do NOT preinstall official catboost (Kaggle's do).
    try:
        import catboost  # noqa: F401
    except ModuleNotFoundError:
        _sh("pip install -q catboost==1.2.10", timeout=1800)

    import catboost as official
    import catboost_rs

    sys.path.insert(0, os.path.join(REPO, "bench"))
    import generator as gen

    result["provenance"]["catboost_version"] = official.__version__
    log(f"official catboost {official.__version__}")

    n_rows, n_features = SHAPE
    X_clean, y = gen.generate(n_rows, n_features, seed=RANDOM_SEED)
    # The NaN pool: one column in five carries NaNs, so the sentinel branch is live on a
    # real fraction of the feature axis rather than in a single column that a border-count
    # accident could make irrelevant.
    X_nan = X_clean.copy()
    rng = np.random.default_rng(RANDOM_SEED)
    for f in range(0, n_features, 5):
        mask = rng.random(n_rows) < 0.10
        X_nan[mask, f] = np.nan

    def device_probe(cell):
        """OBSERVE device activation with CB_GPU_PROF over a 2-iteration fit."""
        probe = os.path.join(WORK, "_probe.py")
        with open(probe, "w") as fh:
            fh.write(
                "import os, sys, json, numpy as np, catboost_rs\n"
                f"kw = json.loads({json.dumps(json.dumps(cell['kwargs']))})\n"
                "kw['iterations'] = 2\n"
                f"sys.path.insert(0, {json.dumps(os.path.join(REPO, 'bench'))})\n"
                "import generator as gen\n"
                f"X, y = gen.generate({n_rows}, {n_features}, seed={RANDOM_SEED})\n"
                f"if {bool(cell['nan_pool'])}:\n"
                f"    rng = np.random.default_rng({RANDOM_SEED})\n"
                f"    for f in range(0, {n_features}, 5):\n"
                f"        X[rng.random({n_rows}) < 0.10, f] = np.nan\n"
                "catboost_rs.CatBoostRegressor(**kw).fit(X, y)\n"
            )
        env = dict(os.environ, CB_GPU_PROF="1")
        rc, out = _sh([sys.executable, probe], env=env, timeout=1800)
        return out.count("CB_GPU_PROF tree"), rc, out[-1500:]

    def time_rs(cell, X):
        m = catboost_rs.CatBoostRegressor(**cell["kwargs"])
        t0 = time.time()
        m.fit(X, y)
        return time.time() - t0

    def time_official(cell, X):
        kw = dict(cell["kwargs"], task_type="GPU", devices="0", verbose=False)
        m = official.CatBoostRegressor(**kw)
        t0 = time.time()
        m.fit(X, y)
        return time.time() - t0

    for cell in build_grid():
        if time.time() - started > BUDGET_S:
            result["errors"]["budget"] = "BUDGET EXCEEDED before " + cell["name"]
            log(result["errors"]["budget"])
            break

        log(f"\n=== {cell['name']} ({cell['group']})")
        X = X_nan if cell["nan_pool"] else X_clean
        row = dict(cell)
        row.pop("kwargs", None)
        row["kwargs"] = cell["kwargs"]

        n_tree_lines, prc, ptail = device_probe(cell)
        row["device_tree_lines"] = n_tree_lines
        row["device_observed"] = n_tree_lines > 0
        if prc != 0:
            row["probe_error"] = ptail
        log(f"  probe: {n_tree_lines} CB_GPU_PROF tree lines (expect_device={cell['expect_device']})")

        # The routing assertion. A violation invalidates the cell's INTERPRETATION, so it
        # is recorded as a harness failure rather than quietly timed.
        if row["device_observed"] != cell["expect_device"]:
            row["harness_failure"] = (
                f"expected device={cell['expect_device']} but observed "
                f"{row['device_observed']} ({n_tree_lines} tree lines) — "
                "device_host_eligible does not match what the routing tests assert"
            )
            log("  HARNESS FAILURE: " + row["harness_failure"])

        rs_times, off_times = [], []
        try:
            for _ in range(REPEATS):
                rs_times.append(time_rs(cell, X))
        except Exception as exc:  # noqa: BLE001 — recorded, not swallowed
            row["rs_error"] = f"{type(exc).__name__}: {exc}"
            log(f"  catboost-rs FAILED: {row['rs_error']}")

        if cell.get("official_na"):
            row["official_na"] = cell["official_na"]
            log(f"  official: N/A — {cell['official_na']}")
        else:
            try:
                for _ in range(REPEATS):
                    off_times.append(time_official(cell, X))
            except Exception as exc:  # noqa: BLE001
                row["official_error"] = f"{type(exc).__name__}: {exc}"
                log(f"  official FAILED: {row['official_error']}")

        def stats(ts):
            if not ts:
                return None
            s = sorted(ts)
            return {"median": s[len(s) // 2], "min": s[0], "max": s[-1]}

        row["rs"] = stats(rs_times)
        row["official"] = stats(off_times)
        if row["rs"] and row["official"]:
            row["ratio_median"] = row["official"]["median"] / row["rs"]["median"]
            lo = row["official"]["min"] / row["rs"]["max"]
            hi = row["official"]["max"] / row["rs"]["min"]
            row["ratio_range"] = [lo, hi]
            row["within_noise"] = lo < 1.0 < hi
        if row["rs"]:
            log(f"  catboost-rs median {row['rs']['median']:.2f}s")
        if row["official"]:
            log(f"  official   median {row['official']['median']:.2f}s")

        result["grid"].append(row)
        json.dump(result, open(os.path.join(WORK, "result.json"), "w"), indent=2)

    write_report(result)
    return 0


def write_report(result):
    L = []
    L.append("# catboost-rs param-wave GPU speed")
    L.append("")
    L.append(f"- GPU: `{result['provenance'].get('gpu', '?')}`")
    L.append(f"- commit: `{result['provenance'].get('commit', '?')}`")
    L.append(f"- official catboost: `{result['provenance'].get('catboost_version', '?')}`")
    L.append(f"- shape: {SHAPE[0]} x {SHAPE[1]}, {ITERS} iterations, depth {DEPTH}, "
             f"border_count {BORDER_COUNT}, {REPEATS} repeats")
    L.append("")

    baseline = next(
        (r for r in result["grid"] if r["name"] == "baseline" and r.get("rs")), None
    )

    L.append("## A. The device-decline cliff")
    L.append("")
    L.append("`model_shrink_rate != 0` and `leaf_estimation_iterations > 1` route the fit to "
             "the CPU grower. Both look like ordinary knobs; this is what setting one costs "
             "on a GPU box.")
    L.append("")
    L.append("| cell | device? | catboost-rs median | vs baseline | official GPU median |")
    L.append("|---|---|---|---|---|")
    for r in result["grid"]:
        if r["group"] != "decline-cliff":
            continue
        rs = f"{r['rs']['median']:.2f}s" if r.get("rs") else (r.get("rs_error") or "—")
        off = f"{r['official']['median']:.2f}s" if r.get("official") else (
            r.get("official_na") or r.get("official_error") or "—")
        rel = "—"
        if baseline and r.get("rs"):
            rel = f"{r['rs']['median'] / baseline['rs']['median']:.2f}x"
        dev = "yes" if r.get("device_observed") else "NO (CPU grower)"
        if r.get("harness_failure"):
            dev += " ** HARNESS FAILURE **"
        L.append(f"| `{r['name']}` | {dev} | {rs} | {rel} | {off} |")
    L.append("")

    L.append("## B. Border-build cost")
    L.append("")
    L.append("| feature_border_type | device? | catboost-rs median | vs GreedyLogSum | "
             "official GPU median |")
    L.append("|---|---|---|---|---|")
    gls = next(
        (r for r in result["grid"]
         if r["name"] == "feature_border_type=GreedyLogSum" and r.get("rs")),
        None,
    )
    for r in result["grid"]:
        if r["group"] != "border-build":
            continue
        rs = f"{r['rs']['median']:.2f}s" if r.get("rs") else (r.get("rs_error") or "—")
        off = f"{r['official']['median']:.2f}s" if r.get("official") else (
            r.get("official_na") or r.get("official_error") or "—")
        rel = f"{r['rs']['median'] / gls['rs']['median']:.2f}x" if (gls and r.get("rs")) else "—"
        dev = "yes" if r.get("device_observed") else "NO"
        L.append(f"| `{r['name'].split('=')[1]}` | {dev} | {rs} | {rel} | {off} |")
    L.append("")

    L.append("## C. nan_mode (control)")
    L.append("")
    L.append("Expected to be free. Measured because it is expected to be free: `Max` adds a "
             "per-object sentinel branch to the quantizer on BOTH the host and the device "
             "kernel, and an assumed-free per-object change is exactly how a regression "
             "ships unnoticed.")
    L.append("")
    L.append("| cell | device? | catboost-rs median | official GPU median |")
    L.append("|---|---|---|---|")
    for r in result["grid"]:
        if r["group"] != "nan-mode":
            continue
        rs = f"{r['rs']['median']:.2f}s" if r.get("rs") else (r.get("rs_error") or "—")
        off = f"{r['official']['median']:.2f}s" if r.get("official") else (
            r.get("official_na") or r.get("official_error") or "—")
        dev = "yes" if r.get("device_observed") else "NO"
        L.append(f"| `{r['name']}` | {dev} | {rs} | {off} |")
    L.append("")

    failures = [r for r in result["grid"] if r.get("harness_failure")]
    if failures:
        L.append("## HARNESS FAILURES")
        L.append("")
        for r in failures:
            L.append(f"- `{r['name']}`: {r['harness_failure']}")
        L.append("")

    L.append("## Disciplines")
    L.append("")
    L.append("- Device activation is OBSERVED per cell via `CB_GPU_PROF` tree lines, in both "
             "directions: a cell expected to commit that shows none, AND a cell expected to "
             "decline that shows some, are both harness failures.")
    L.append("- Both sides get the same explicit recipe; a recipe official CatBoost GPU "
             "cannot express is `N/A` with the reason, never swapped for another.")
    L.append("- Median/min/max over repeats; a ratio range spanning 1.0 is *within noise*.")
    L.append("- A failed build or cell yields an error row, never an invented number.")
    with open(os.path.join(WORK, "report.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    if "--dry-run" in sys.argv:
        sys.exit(dry_run())
    sys.exit(main())
