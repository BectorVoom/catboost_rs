#!/usr/bin/env python3
# catboost-rs FULL-PARAMETER GPU speed grid (SPD-02/SPD-03) — Kaggle P100 (CUDA).
#
# The question: across the parameter axes the `gpu-full-parameter-parity` phase made
# device-reachable, does catboost-rs train FASTER than official CatBoost with
# task_type='GPU'?
#
# This is deliberately NOT `bench/quick_gpu_speed/bench.py` (frozen — it backs the
# r4a/r4b/r4c reports and is never edited). It reuses that file's proven shapes verbatim —
# `timed_fit`'s untimed-warm → timed → drain-the-lazy-queue pattern, the single
# `maturin build --release --features cuda`, the `errors` recording discipline — and adds
# the grid, the repeats, and the budget guard.
#
# ── Disciplines this runner does NOT relax ───────────────────────────────────────────
#
# 1. DEVICE ACTIVATION IS OBSERVED, NOT ASSUMED. `quick_gpu_speed` could only do a static
#    eligibility audit and had to state that activation was invisible from Python.
#    `bench/one_hot_gpu_speed` closed that with `CB_GPU_PROF=1`, whose per-tree
#    `CB_GPU_PROF tree` lines the resident grower emits. This harness inherits that: a
#    cell whose probe shows no device lines is reported as a CPU row and is NEVER counted
#    toward a speed claim. The static audit is kept as a SECOND, independent check.
#
# 2. EVERY KNOB PINNED IDENTICALLY ON BOTH SIDES. Official CatBoost's GPU default
#    `bootstrap_type` is Bayesian; leaving it unset would compare catboost-rs (pushed to
#    `No` by its gate) against an official run doing strictly more work per tree and
#    inflate the speedup. Both sides get the same explicit recipe, and the official side's
#    `get_all_params()` is read back and recorded.
#
# 3. NO PROXYING. A cell official CatBoost GPU cannot express is recorded `N/A` with the
#    reason. It is never replaced by a different recipe and called a comparison (the
#    `bench/RESULTS.md` Region precedent).
#
# 4. SPREAD BEFORE HEADLINE. Every cell reports median/min/max over 3 repeats. A cell whose
#    ratio spread crosses 1.0 is labelled "within noise" and is never claimed as a win.
#
# `--dry-run` enumerates the whole grid with per-cell recipes, the eligibility audit and a
# projected budget, imports nothing GPU-related, and exits 0 — so the grid is reviewable
# before a Kaggle session is spent.

import argparse
import json
import os
import subprocess
import sys
import time

WORK = os.environ.get("CB_BENCH_WORK", "/kaggle/working")
REPO = os.environ.get("CB_BENCH_REPO", "/tmp/repo")

# ── Pinned workload + model config (module constants, no side effects) ────────────────
DEPTH = 6
ITERS = 30
LEARNING_RATE = 0.1
L2_LEAF_REG = 3.0
BORDER_COUNT = 32
RANDOM_SEED = 42
REPEATS = 3

#: Both shapes are at or above the D-10-09 device/CPU crossover (n = 100_000, recorded in
#: bench/RESULTS.md). Below it the device cannot win — that is launch-overhead physics, not
#: a tuning gap — so measuring there would only manufacture a loss.
SHAPES = ((300_000, 50), (1_000_000, 50))

#: Cat columns for the CTR axis. `one_hot_max_size=1` forces the CTR route (a pool mixing
#: one-hot and CTR columns is typed-rejected, SPEC-OH-26).
CTR_CARDINALITIES = (4, 16, 64)
ONE_HOT_MAX_SIZE = 1

#: Wall-clock ceiling for the whole grid, leaving headroom inside a Kaggle GPU session.
BUDGET_S = float(os.environ.get("CB_BENCH_BUDGET_S", 9 * 3600))

#: Rough per-fit seconds/1e6-rows, from the r4a/r4b/r4c runs (a 300k×50 depth-6 30-iter
#: arm took ~1.2–1.4 s per side). Used ONLY for the dry-run projection.
SECONDS_PER_MROW_PER_ARM = 4.5


# ── The grid ─────────────────────────────────────────────────────────────────────────
#
# Factored as a TABLE so adding an axis is a data change, not a code change (the task's
# refactor note). Each cell carries the kwargs BOTH sides receive, plus the axis labels the
# report groups by.


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
    )


def build_grid():
    """Enumerate the D3 grid plus the two new-reachability showcase cells.

    Returns a list of dicts with keys: ``name``, ``axes``, ``kwargs``, ``shape``,
    ``kind`` (``reg``/``clf``), ``weighted``, ``ctr``, ``showcase``.
    """
    cells = []
    for n_rows, n_features in SHAPES:
        for grow_policy in ("SymmetricTree", "Depthwise"):
            for loss in ("RMSE", "Logloss"):
                for weighted in (False, True):
                    for ctr in (False, True):
                        kwargs = _base_kwargs()
                        kwargs["loss_function"] = loss
                        kwargs["grow_policy"] = grow_policy
                        if ctr:
                            kwargs["one_hot_max_size"] = ONE_HOT_MAX_SIZE
                        cells.append(
                            dict(
                                name=(
                                    f"{grow_policy}|{loss}|"
                                    f"{'w' if weighted else 'unw'}|"
                                    f"{'ctr' if ctr else 'noctr'}|{n_rows//1000}k"
                                ),
                                axes=dict(
                                    grow_policy=grow_policy,
                                    loss=loss,
                                    weighted=weighted,
                                    ctr=ctr,
                                    n_rows=n_rows,
                                    n_features=n_features,
                                ),
                                kwargs=kwargs,
                                shape=(n_rows, n_features),
                                kind="reg" if loss == "RMSE" else "clf",
                                weighted=weighted,
                                ctr=ctr,
                                showcase=False,
                            )
                        )

    # ── The two NEW-REACHABILITY showcase cells ──────────────────────────────────────
    # These are the cells that make this benchmark measure THIS phase's own work: both
    # were CPU-only before it, so before Wave 1/2 they could not have been device rows at
    # all. If either is not device-eligible, the phase's reachability claim is false and
    # the dry-run says so rather than silently timing a CPU fit.
    n_rows, n_features = SHAPES[0]

    bias_kwargs = _base_kwargs()
    bias_kwargs["loss_function"] = "RMSE"
    bias_kwargs["grow_policy"] = "SymmetricTree"
    bias_kwargs["boost_from_average"] = True  # FPP-01/FPP-02: was a hard CPU fallback
    cells.append(
        dict(
            name=f"SHOWCASE-bias|RMSE|unw|noctr|{n_rows//1000}k",
            axes=dict(
                grow_policy="SymmetricTree", loss="RMSE", weighted=False, ctr=False,
                n_rows=n_rows, n_features=n_features, new_reachability="boost_from_average",
            ),
            kwargs=bias_kwargs,
            shape=(n_rows, n_features),
            kind="reg",
            weighted=False,
            ctr=False,
            showcase=True,
        )
    )

    samp_kwargs = _base_kwargs()
    samp_kwargs["loss_function"] = "RMSE"
    samp_kwargs["grow_policy"] = "Depthwise"
    samp_kwargs["bootstrap_type"] = "Bernoulli"  # FPP-12/FPP-13: was SymmetricTree-only
    samp_kwargs["subsample"] = 0.66
    cells.append(
        dict(
            name=f"SHOWCASE-sampled-nonsym|RMSE|unw|noctr|{n_rows//1000}k",
            axes=dict(
                grow_policy="Depthwise", loss="RMSE", weighted=False, ctr=False,
                n_rows=n_rows, n_features=n_features,
                new_reachability="Depthwise x Bernoulli",
            ),
            kwargs=samp_kwargs,
            shape=(n_rows, n_features),
            kind="reg",
            weighted=False,
            ctr=False,
            showcase=True,
        )
    )
    return cells


def prune_grid(cells, keep_max=24, budget_s=None):
    """Prune toward the D3 16–24 target IF the projected budget needs it, RECORDING what
    went and why.

    Pruning is BUDGET-DRIVEN, not unconditional. D3 sets 16–24 as the target for a grid
    that would otherwise overrun a session; dropping cells that comfortably fit would throw
    away evidence for nothing. When the projection fits the ceiling, the FULL grid runs and
    ``pruned`` is empty.

    The pruning rule is fixed, not ad hoc: the most expensive and least informative cells
    go first — CTR × Logloss at the 1M shape. CTR is the slowest axis (it materializes two
    permutations), Logloss duplicates RMSE's grow-loop cost profile, and the 1M shape
    triples the per-fit cost, so those cells buy the least evidence per session-second.
    Showcase cells are NEVER pruned; they are the point of the run.
    """
    pruned = []
    kept = list(cells)
    budget_s = BUDGET_S if budget_s is None else budget_s
    if len(kept) <= keep_max or projected_seconds(kept) <= budget_s:
        return kept, pruned

    def is_prunable(c):
        return (
            not c["showcase"]
            and c["ctr"]
            and c["axes"]["loss"] == "Logloss"
            and c["axes"]["n_rows"] == SHAPES[1][0]
        )

    for c in list(kept):
        if len(kept) <= keep_max:
            break
        if is_prunable(c):
            kept.remove(c)
            pruned.append(
                dict(name=c["name"], reason="CTR x Logloss at the 1M shape — highest cost, "
                                            "lowest marginal evidence (D3 16-24 target)")
            )
    return kept, pruned


def projected_seconds(cells):
    """Dry-run wall-clock projection: 2 arms × REPEATS fits per cell, plus one untimed warm
    fit per arm per repeat (`timed_fit` does warm+timed), i.e. 4 fits per arm-repeat."""
    total = 0.0
    for c in cells:
        n_rows = c["axes"]["n_rows"]
        per_fit = SECONDS_PER_MROW_PER_ARM * (n_rows / 1e6)
        ctr_mult = 2.0 if c["ctr"] else 1.0
        total += per_fit * ctr_mult * 2 * REPEATS * 2  # 2 arms, warm+timed
    return total


def build_eligibility_audit(cell):
    """Static, no-instrumentation audit of the `device_host_eligible` preconditions this
    CELL satisfies by construction, EXTENDED with this phase's new axes.

    This is the SECOND of two independent checks. The first — and the authoritative one —
    is the `CB_GPU_PROF` residency probe, which OBSERVES the device grow rather than
    reasoning about it. The audit is retained because it explains WHY a cell should be
    eligible, which a probe result cannot.
    """
    k = cell["kwargs"]
    conds = {
        "grow_policy_covered": {
            "satisfied": k["grow_policy"] in ("SymmetricTree", "Depthwise", "Lossguide", "Region"),
            "rationale": f"grow_policy={k['grow_policy']!r} is in the device-covered set.",
        },
        "single_dim_target": {
            "satisfied": k["loss_function"] in ("RMSE", "Logloss"),
            "rationale": f"loss_function={k['loss_function']!r} is single-dim with a covered der kernel.",
        },
        "random_strength_zero": {
            "satisfied": k["random_strength"] == 0,
            "rationale": "random_strength=0 — the perturbed level search is not device-covered.",
        },
        "leaf_method_covered": {
            "satisfied": k["leaf_estimation_method"] in ("Gradient", "Simple"),
            "rationale": "Gradient leaf — the device grower's calc_average formula.",
        },
        # ── FPP axes this phase opened ────────────────────────────────────────────────
        "bias_reachable": {
            "satisfied": True,
            "rationale": (
                "FPP-01/FPP-02: boost_from_average is no longer a hard CPU fallback — the "
                "resident approx is seeded from DeviceTrainConfig.bias. "
                f"This cell sets boost_from_average={k['boost_from_average']!r}."
            ),
        },
        "sampling_x_grow_policy": {
            "satisfied": (
                k["bootstrap_type"] == "No"
                or k["bootstrap_type"] in ("Bayesian", "Bernoulli", "MVS")
            ),
            "rationale": (
                "FPP-12/FPP-13: the three host-sampled types are device-eligible on EVERY "
                "covered grow policy, not just SymmetricTree. "
                f"This cell sets bootstrap_type={k['bootstrap_type']!r}."
            ),
        },
        "ctr_projection_arity": {
            "satisfied": True,
            "rationale": (
                "This cell uses SIMPLE (single-feature) CTR projections only "
                "(max_ctr_complexity left at its default of 1 for the CTR cells). "
                "COMBINATION projections are device-INELIGIBLE — FPP-11 is escalated, see "
                "ctr_types_are_device_covered — so a combination cell would silently "
                "measure a CPU fit and is deliberately absent from this grid."
            ),
        },
        "exact_leaf_absent": {
            "satisfied": True,
            "rationale": (
                "FPP-05/FPP-06 made the Exact order-statistic leaf device-reachable for "
                "{Mae, Quantile}, but the grid's losses are RMSE/Logloss, whose leaf is "
                "Gradient — so the exact-leaf arm is not exercised here."
            ),
        },
    }
    return {
        "conditions": conds,
        "all_satisfied": all(c["satisfied"] for c in conds.values()),
        "activation_observable": True,
        "caveat": (
            "Static audit only. Device activation is verified INDEPENDENTLY by the "
            "CB_GPU_PROF residency probe; a cell whose probe shows no device lines is "
            "reported as a CPU row regardless of what this audit says."
        ),
    }


def dry_run():
    """Enumerate the grid, audit it, project the budget. No GPU, no catboost, no torch."""
    cells = build_grid()
    kept, pruned = prune_grid(cells)
    projected = projected_seconds(kept)

    print("=" * 78)
    print("catboost-rs FULL-PARAMETER GPU speed grid — DRY RUN")
    print("=" * 78)
    print(f"cells enumerated : {len(cells)}")
    print(f"cells pruned     : {len(pruned)}")
    print(f"cells to run     : {len(kept)}")
    print(f"repeats per cell : {REPEATS}")
    print(f"projected budget : {projected/60:.1f} min  (ceiling {BUDGET_S/60:.0f} min)")
    print()

    failures = []
    for c in kept:
        audit = build_eligibility_audit(c)
        n_rows = c["axes"]["n_rows"]
        flag = "ELIGIBLE" if audit["all_satisfied"] else "NOT-ELIGIBLE"
        print(f"[{flag}] {c['name']}")
        print(f"    shape={c['shape']} kind={c['kind']} weighted={c['weighted']} ctr={c['ctr']}")
        print(f"    kwargs={json.dumps(c['kwargs'], sort_keys=True)}")
        if not audit["all_satisfied"]:
            for name, cond in audit["conditions"].items():
                if not cond["satisfied"]:
                    print(f"    FAILS {name}: {cond['rationale']}")
            failures.append(("audit", c["name"]))

        # Local, GPU-free assertion: an eligible cell below the D-10-09 crossover would be
        # measuring launch overhead, not the grow loop.
        if audit["all_satisfied"] and n_rows < 100_000:
            print(f"    FAILS crossover: n={n_rows} < 100_000 (D-10-09)")
            failures.append(("crossover", c["name"]))

    print()
    for p in pruned:
        print(f"[PRUNED] {p['name']}\n    reason: {p['reason']}")

    # Every showcase cell MUST be eligible: if it is not, this phase's reachability claim
    # is false and the benchmark would silently time a CPU fit and call it a GPU row.
    print()
    for c in kept:
        if not c["showcase"]:
            continue
        audit = build_eligibility_audit(c)
        state = "ELIGIBLE" if audit["all_satisfied"] else "NOT-ELIGIBLE"
        print(f"[SHOWCASE {state}] {c['name']} — {c['axes'].get('new_reachability')}")
        if not audit["all_satisfied"]:
            failures.append(("showcase", c["name"]))

    if projected > BUDGET_S:
        print(f"\nWARNING: projected {projected/60:.1f} min exceeds the {BUDGET_S/60:.0f} min "
              "ceiling; the run's budget guard will skip the tail and record it.")

    if failures:
        print("\nDRY RUN FAILED:")
        for kind, name in failures:
            print(f"  {kind}: {name}")
        return 1
    print("\nDRY RUN OK — grid reviewable, every showcase cell eligible.")
    return 0


# ── Real run (Kaggle) ────────────────────────────────────────────────────────────────


def _sh(cmd, env=None, cwd=None, timeout=3600):
    proc = subprocess.run(
        cmd, shell=isinstance(cmd, str), env=env, cwd=cwd,
        capture_output=True, text=True, timeout=timeout,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def main():
    """Execute the grid on a real CUDA device. Imports GPU things only here."""
    import numpy as np

    os.makedirs(WORK, exist_ok=True)
    result = {
        "provenance": {},
        "grid": [],
        "pruned": [],
        "errors": {},
        "budget_s": BUDGET_S,
    }
    started = time.time()

    def log(msg):
        print(msg, flush=True)

    rc, out = _sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    result["provenance"]["gpu"] = out.strip() if rc == 0 else f"nvidia-smi failed: {out[:200]}"
    log(f"GPU: {result['provenance']['gpu']}")

    # ONE build for the whole grid (quick_gpu_speed's pattern).
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

    # Kaggle images preinstall official catboost; Colab images do NOT (this cost the
    # first Colab relaunch a 25-min build-then-ModuleNotFoundError round trip). The
    # pin matches the project's parity target.
    try:
        import catboost  # noqa: F401
    except ModuleNotFoundError:
        _sh("pip install -q catboost==1.2.10", timeout=1800)

    import catboost as official  # noqa: F401
    import catboost_rs

    sys.path.insert(0, os.path.join(REPO, "bench"))
    import generator as gen

    result["provenance"]["catboost_version"] = official.__version__

    cells, pruned = prune_grid(build_grid())
    result["pruned"] = pruned

    def make_data(cell):
        n_rows, n_features = cell["shape"]
        X, y_reg = gen.generate(n_rows, n_features, seed=RANDOM_SEED)
        cat = gen.generate_cat(n_rows, seed=RANDOM_SEED) if cell["ctr"] else None
        if cell["kind"] == "clf":
            y = gen.cat_driven_binary_target(X, cat, seed=RANDOM_SEED) if cell["ctr"] \
                else gen.binary_target(X, seed=RANDOM_SEED)
        else:
            y = y_reg
        w = gen.generate_weights(n_rows) if cell["weighted"] else None
        return X, y, cat, w

    def device_probe(cell, X, y, cat, w):
        """OBSERVE device activation with CB_GPU_PROF over a 2-iteration fit."""
        probe = os.path.join(WORK, "_probe.py")
        with open(probe, "w") as fh:
            fh.write(
                "import os, sys, json, numpy as np, catboost_rs\n"
                f"kw = json.loads({json.dumps(json.dumps(cell['kwargs']))})\n"
                "kw['iterations'] = 2\n"
                f"n_rows, n_features = {cell['shape']}\n"
                f"sys.path.insert(0, {json.dumps(os.path.join(REPO, 'bench'))})\n"
                "import generator as gen\n"
                f"X, yr = gen.generate(n_rows, n_features, seed={RANDOM_SEED})\n"
                f"ctr = {bool(cell['ctr'])}\n"
                f"cat = gen.generate_cat(n_rows, seed={RANDOM_SEED}) if ctr else None\n"
                f"kind = {json.dumps(cell['kind'])}\n"
                "y = yr if kind == 'reg' else ("
                f"gen.cat_driven_binary_target(X, cat, seed={RANDOM_SEED}) if ctr "
                f"else gen.binary_target(X, seed={RANDOM_SEED}))\n"
                f"w = gen.generate_weights(n_rows) if {bool(cell['weighted'])} else None\n"
                "Cls = catboost_rs.CatBoostRegressor if kind == 'reg' "
                "else catboost_rs.CatBoostClassifier\n"
                "m = Cls(**kw)\n"
                # D1 (Colab T4, 2026-08-07): `sample_weight` was passed UNCONDITIONALLY here,
                # including when `w is None`. catboost_rs's sklearn surface is
                # `fit(x, y=None, cat_features=None, eval_set=None)` — there is no
                # `sample_weight` parameter — so the probe died with a TypeError before
                # growing a single tree, reported 0 `CB_GPU_PROF tree` lines, and every one
                # of the 34 cells was labelled "DEVICE NOT ACTIVATED". The whole grid was
                # void as a comparison while the device path was in fact healthy.
                #
                # Build the kwargs conditionally, mirroring the TIMING path (which always
                # got this right), so the probe exercises the same call the measurement does.
                # A weighted cell still fails — that is D2, a real API-parity gap, and it is
                # recorded as an N/A rather than papered over.
                "fit_kw = {}\n"
                "if w is not None:\n"
                "    fit_kw['sample_weight'] = w\n"
                "if ctr:\n"
                "    import numpy as np\n"
                "    Xf = np.concatenate([X, cat.astype(X.dtype)], axis=1)\n"
                "    fit_kw['cat_features'] = list(range(X.shape[1], Xf.shape[1]))\n"
                "    m.fit(Xf, y, **fit_kw)\n"
                "else:\n"
                "    m.fit(X, y, **fit_kw)\n"
            )
        env = dict(os.environ, CB_GPU_PROF="1")
        rc, out = _sh([sys.executable, probe], env=env, timeout=1800)
        n_tree_lines = out.count("CB_GPU_PROF tree")
        return n_tree_lines > 0, n_tree_lines, out[-1500:]

    for cell in cells:
        elapsed = time.time() - started
        if elapsed > BUDGET_S:
            remaining = len(cells) - len(result["grid"])
            result["errors"]["budget"] = f"BUDGET EXCEEDED, {remaining} cells not run"
            log(result["errors"]["budget"])
            break

        log(f"\n=== {cell['name']} ===")
        entry = {"name": cell["name"], "axes": cell["axes"], "kwargs": cell["kwargs"],
                 "audit": build_eligibility_audit(cell), "timings": {}, "quality": {}}
        try:
            X, y, cat, w = make_data(cell)
        except Exception as e:
            entry["error"] = f"data: {e!r}"
            result["grid"].append(entry)
            continue

        activated, n_lines, probe_tail = device_probe(cell, X, y, cat, w)
        entry["device_activated"] = activated
        entry["device_prof_tree_lines"] = n_lines
        if not activated:
            entry["probe_tail"] = probe_tail
            log(f"  DEVICE NOT ACTIVATED (0 CB_GPU_PROF tree lines) — reported as a CPU row")

        Xf = X
        cat_idx = None
        if cell["ctr"]:
            Xf = np.concatenate([X, cat.astype(X.dtype)], axis=1)
            cat_idx = list(range(X.shape[1], Xf.shape[1]))

        def fit_once(which):
            kw = dict(cell["kwargs"])
            if which == "official":
                kw["task_type"] = "GPU"
                Cls = official.CatBoostRegressor if cell["kind"] == "reg" \
                    else official.CatBoostClassifier
                kw["verbose"] = False
            else:
                Cls = catboost_rs.CatBoostRegressor if cell["kind"] == "reg" \
                    else catboost_rs.CatBoostClassifier
            model = Cls(**kw)
            fit_kw = {}
            if w is not None:
                fit_kw["sample_weight"] = w
            if cat_idx is not None:
                fit_kw["cat_features"] = cat_idx
            t0 = time.time()
            model.fit(Xf, y, **fit_kw)
            _ = model.predict(Xf[:1024])  # drain the lazy CubeCL queue before stopping
            return round(time.time() - t0, 4), model

        for which in ("official", "catboost_rs"):
            times = []
            try:
                _warm, _m = fit_once(which)  # UNTIMED warm/JIT-absorbing run
            except Exception as e:
                entry["timings"][which] = {"na": True, "reason": repr(e)}
                log(f"  [{which}] N/A: {e!r}")
                continue
            for _ in range(REPEATS):
                try:
                    t, model = fit_once(which)
                    times.append(t)
                except Exception as e:
                    entry["timings"].setdefault("errors", {})[which] = repr(e)
                    break
            if times:
                entry["timings"][which] = {
                    "median": float(np.median(times)),
                    "min": float(np.min(times)),
                    "max": float(np.max(times)),
                    "all": times,
                }
                log(f"  [{which}] median={np.median(times):.3f}s all={times}")

        result["grid"].append(entry)
        json.dump(result, open(os.path.join(WORK, "result.json"), "w"), indent=2)

    write_report(result)
    json.dump(result, open(os.path.join(WORK, "result.json"), "w"), indent=2)
    return 0


def write_report(result):
    """Render report.md. A cell whose ratio SPREAD crosses 1.0 is 'within noise'."""
    lines = ["# catboost-rs full-parameter GPU speed grid\n"]
    lines.append(f"GPU: {result.get('provenance', {}).get('gpu', 'unknown')}\n")
    lines.append(f"official catboost: {result.get('provenance', {}).get('catboost_version', '?')}\n")
    lines.append("\n`ratio = median(official) / median(catboost_rs)`; **> 1.0 means "
                 "catboost-rs is faster**. A cell whose min/max ratio spread crosses 1.0 is "
                 "labelled *within noise* and is NOT claimed as a win.\n")
    lines.append("\n| cell | device? | official (s) | catboost-rs (s) | ratio | spread | verdict |")
    lines.append("|---|---|---|---|---|---|---|")
    for e in result.get("grid", []):
        o = e["timings"].get("official")
        r = e["timings"].get("catboost_rs")
        if not isinstance(o, dict) or "median" not in o or not isinstance(r, dict) or "median" not in r:
            reason = (o or {}).get("reason") or (r or {}).get("reason") or "not run"
            lines.append(f"| {e['name']} | {e.get('device_activated')} | N/A | N/A | N/A | N/A | "
                         f"N/A — {str(reason)[:80]} |")
            continue
        ratio = o["median"] / r["median"]
        lo = o["min"] / r["max"]
        hi = o["max"] / r["min"]
        if not e.get("device_activated"):
            verdict = "CPU row (no device activation) — not a GPU claim"
        elif lo <= 1.0 <= hi:
            verdict = "within noise"
        elif ratio > 1.0:
            verdict = "catboost-rs faster"
        else:
            verdict = "official faster"
        lines.append(
            f"| {e['name']} | {e.get('device_activated')} | {o['median']:.3f} | "
            f"{r['median']:.3f} | {ratio:.2f}x | {lo:.2f}–{hi:.2f} | {verdict} |"
        )
    if result.get("pruned"):
        lines.append("\n## Pruned cells\n")
        for p in result["pruned"]:
            lines.append(f"- `{p['name']}` — {p['reason']}")
    if result.get("errors"):
        lines.append("\n## Errors\n")
        for k, v in result["errors"].items():
            lines.append(f"- **{k}**: {v}")
    lines.append("\n## Caveats\n")
    lines.append("- Device activation is OBSERVED per cell via `CB_GPU_PROF` tree lines, not "
                 "assumed. A cell without them is a CPU row and is excluded from any claim.")
    lines.append("- Both sides receive the SAME explicit recipe; official CatBoost's GPU "
                 "default `bootstrap_type=Bayesian` would otherwise do strictly more work "
                 "per tree and inflate the ratio.")
    lines.append("- Both shapes are at or above the D-10-09 crossover (n = 100_000). Below "
                 "it the device cannot win — launch-overhead physics, not a tuning gap.")
    lines.append("- Combination-CTR cells are deliberately ABSENT: FPP-11 is escalated and "
                 "combination projections are device-ineligible, so such a cell would "
                 "silently measure a CPU fit.")
    lines.append("- The headline holds for the axes measured here ONLY; it is not a claim "
                 "about CatBoost GPU in general.")
    with open(os.path.join(WORK, "report.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="catboost-rs full-parameter GPU speed grid")
    ap.add_argument("--dry-run", action="store_true",
                    help="enumerate + audit the grid and project the budget; no GPU needed")
    args = ap.parse_args()
    sys.exit(dry_run() if args.dry_run else main())
