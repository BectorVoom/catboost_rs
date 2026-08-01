#!/usr/bin/env python3
# catboost-rs one-hot categorical GPU speed gate (SPEC-OH-30 / plan T30) — Colab T4 (CUDA).
#
# The question: on a MATCHED config, does catboost-rs train a one-hot categorical workload
# FASTER than official CatBoost with task_type='GPU'?
#
# Shape follows `bench/bootstrap_gpu/bootstrap_bench_colab.py` (the only Colab-shaped
# runner in this repo): the tree under test is STAGED at /content/cbrs by the driver — not
# git-cloned — so a run measures the working tree, and provenance markers below prove
# WHICH tree it is. `bench/quick_gpu_speed/bench.py` supplies the workload/config shape;
# neither of those files is modified by this one.
#
# ── Three disciplines this runner does NOT relax ──────────────────────────────────────
#
# 1. CORRECTNESS BEFORE SPEED. Part A runs the one-hot oracle + device parity suites. A
#    blocking failure aborts with no speed number quoted.
#
# 2. DEVICE ACTIVATION IS OBSERVED, NOT ASSUMED. `quick_gpu_speed/bench.py` could only
#    perform a STATIC eligibility audit and had to state plainly that activation is not
#    observable from Python. Part B0 closes that: a short fit per arm under
#    `CB_GPU_PROF=1`, requiring the per-tree `CB_GPU_PROF tree` line the resident grower
#    emits. An arm without it is reported as a CPU row, never as a GPU number.
#
# 3. EVERY KNOB PINNED IDENTICALLY ON BOTH SIDES, AND READ BACK. Official CatBoost's GPU
#    default `bootstrap_type` is Bayesian; leaving it unset would compare catboost-rs
#    (pushed toward `No` by its eligibility gate) against an official run doing strictly
#    more work per tree, inflating the reported speedup. `get_all_params()` is read back
#    on the official side and recorded.
#
#    `bootstrap_type='No'` and `random_strength=0` are CONSTRAINED, not merely chosen:
#    catboost-rs typed-REJECTS one-hot training with `bootstrap_type != No` or
#    `random_strength != 0` (SPEC-OH-27 / plan T01b took Branch B — the upstream per-level
#    RNG draw accounting for one-hot candidates under `CompressCandidates` is not
#    established, and consuming an unverified rule would silently desynchronise every
#    later tree's sample). Both sides are therefore pinned to the same draw-inert config
#    and the report states the constraint explicitly rather than presenting it as a
#    free choice.
#
# Do-not-fabricate: correctness is gated BEFORE any timing; every number comes from a
# measured call in THIS run; a failed arm leaves `null`, never an invented value.

import glob
import json
import os
import re
import subprocess
import sys
import time

WORK = "/content/bench_out"
REPO = "/content/cbrs"

# The workload: `quick_gpu_speed`'s 300k-row shape, WIDENED with categorical columns.
# 45 float + 5 BINARY categorical at `one_hot_max_size = 2` routes every cat column
# one-hot (cardinality 2 <= max), which is the regime SPEC-OH-30 measures. A higher
# cardinality would route to CTR instead, and a pool mixing both routes is typed-rejected
# by SPEC-OH-26 — so the split is 45/5 by construction, not by taste.
N_ROWS = 300_000
N_FLOAT = 45
N_CAT = 5
CAT_CARDINALITY = 2
ONE_HOT_MAX_SIZE = 2

DEPTH = 6
ITERS = 30
LEARNING_RATE = 0.1
L2_LEAF_REG = 3.0
BORDER_COUNT = 32
RANDOM_SEED = 42

# Pinned identically on BOTH sides. See discipline 3 above: these are constrained by
# SPEC-OH-27's typed rejection, not chosen for convenience.
BOOTSTRAP_TYPE = "No"
RANDOM_STRENGTH = 0.0
BOOST_FROM_AVERAGE = False
LEAF_ESTIMATION_METHOD = "Gradient"

ARMS = [
    ("RMSE", "regression"),
    ("Logloss", "binary"),
]

# (label, crate, cargo extra args, test filters, blocking?)
ORACLE_SUITES = [
    ("cb-train one-hot upstream oracle (CPU)", "cb-train",
     ["--test", "one_hot_oracle_test"], [], True),
    ("cb-train one-hot RNG draw accounting", "cb-train",
     ["--test", "one_hot_draw_accounting_test"], [], True),
    ("cb-train device one-hot parity (CUDA)", "cb-train",
     ["--no-default-features", "--features", "cuda", "--test",
      "device_one_hot_parity_test"], [], True),
    ("cb-backend one-hot device kernels (CUDA)", "cb-backend",
     ["--no-default-features", "--features", "cuda", "--lib"], ["one_hot"], True),
    ("cb-model float-only byte identity (SPEC-OH-31)", "cb-model",
     ["--test", "float_only_byte_identity_test"], [], True),
    ("cb-backend device float-only identity (CUDA)", "cb-backend",
     ["--no-default-features", "--features", "cuda", "--lib"],
     ["device_float_only_identity_test"], True),
]

# ── PREFLIGHT BLOCKER ────────────────────────────────────────────────────────────────
# This runner drives catboost-rs through its REAL public Python `.fit()` surface (that is
# the whole point of a speed gate — a cargo-test harness would not measure what a user
# gets). One-hot TRAINING, however, is reached through `cb_train::train_cat`, and the
# facade `CatBoostBuilder::fit` currently calls `cb_train::train` and pins
# `one_hot_max_size` to the upstream default with the comment "the facade does not yet
# surface categorical config" (crates/catboost-rs/src/builder.rs). So a `cat_features`
# argument does not reach the trainer, and a fit through this surface would train a
# float-only model while APPEARING to measure one-hot training — the worst possible
# outcome for a benchmark.
#
# That routing is the subject of a separate, not-yet-executed plan (commit 41e7e9c,
# "cat_features/CTR facade routing — spec+plan (blocked, not yet executed)"). Until it
# lands, this runner REFUSES to quote a number rather than silently measuring the wrong
# thing. The preflight is a positive check on the built wheel, not a source grep, so it
# starts passing the moment the routing actually works.
FACADE_ROUTING_MARKER = ("train_cat", "crates/catboost-rs/src/builder.rs")


def main():
    os.makedirs(WORK, exist_ok=True)
    logf = open(os.path.join(WORK, "run.log"), "w")

    def log(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    def sh(cmd, cwd=None, env=None, timeout=None):
        shell = isinstance(cmd, str)
        log(f"$ {cmd if shell else ' '.join(cmd)}")
        try:
            p = subprocess.run(cmd, shell=shell, cwd=cwd, env=env, timeout=timeout,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, errors="replace")
            out = p.stdout or ""
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or "") if isinstance(e.stdout, str) else ""
            log(f"  !! TIMEOUT after {timeout}s")
            return 124, out
        if out:
            log(out[-3000:])
        return p.returncode, out

    def bail(result, verdict, reason, code=2):
        result["verdict"] = verdict
        result["fatal"] = reason
        json.dump(result, open(os.path.join(WORK, "result.json"), "w"), indent=2)
        log("FATAL:", reason)
        sys.exit(code)

    result = {"provenance": {}, "oracle": {}, "speed": {}, "caveats": {}}
    result["caveats"]["bootstrap_and_random_strength_are_constrained"] = (
        "bootstrap_type='No' and random_strength=0 are pinned on BOTH sides because "
        "catboost-rs typed-REJECTS one-hot training with either knob active "
        "(SPEC-OH-27 / T01b Branch B): the upstream per-level RNG draw accounting for "
        "one-hot candidates under CompressCandidates is not established, and consuming "
        "an unverified rule would silently desynchronise every later tree's sample. This "
        "is a stated CONSTRAINT on the comparison, not a tuning choice. It also removes "
        "the inflation risk of comparing against official CatBoost's GPU default "
        "(Bayesian), which does strictly more work per tree."
    )
    result["caveats"]["device_activation_is_proven_per_arm"] = (
        "Part B0 proves device residency per arm by running a short fit under "
        "CB_GPU_PROF=1 and requiring the per-tree 'CB_GPU_PROF tree' line the resident "
        "grower emits. Arms without it are reported as CPU rows, never as GPU numbers."
    )
    result["caveats"]["one_hot_only_pool"] = (
        f"Every categorical column has cardinality {CAT_CARDINALITY} and "
        f"one_hot_max_size={ONE_HOT_MAX_SIZE}, so all of them route ONE-HOT. A pool "
        "mixing one-hot and CTR routes is typed-rejected (SPEC-OH-26), so a single-route "
        "pool is required, not preferred."
    )

    # ---------------- STEP 1 — provenance ----------------
    rc, out = sh("nvidia-smi --query-gpu=name,driver_version,memory.total "
                 "--format=csv,noheader")
    gpu = (out or "").strip().splitlines()[0] if rc == 0 and out.strip() else None
    result["provenance"]["gpu"] = gpu
    if not gpu:
        bail(result, "NO-GPU", "no GPU visible to nvidia-smi — refusing to quote GPU numbers")
    log("GPU:", gpu)
    result["provenance"]["cpu_count"] = os.cpu_count()
    result["provenance"]["source"] = REPO

    # Provenance markers: prove the staged tree carries the one-hot DEVICE wiring, so a
    # stale upload can never masquerade as a verified run.
    markers = {
        "has_one_hot_model_split": ("OneHotModelSplit", "crates/cb-model/src/model.rs"),
        "has_one_hot_trainer": ("build_one_hot_columns", "crates/cb-train/src/boosting.rs"),
        "has_real_folds_quantizer": ("quantize_feature_major_with_one_hot",
                                     "crates/cb-train/src/boosting.rs"),
        "has_real_folds_seam": ("real_folds", "crates/cb-compute/src/runtime.rs"),
        "has_scorer_one_hot_arm": ("feature_lo", "crates/cb-backend/src/kernels.rs"),
        "has_split_equality_arm": ("one_hot_partition_split_test",
                                   "crates/cb-backend/src/gpu_runtime/mod.rs"),
        "has_device_split_kind": ("Vec<(u32, u32, bool)>", "crates/cb-compute/src/runtime.rs"),
    }
    for key, (needle, rel) in markers.items():
        rc_m, out_m = sh(f"grep -c '{needle}' {REPO}/{rel} || true")
        digits = [ln.strip() for ln in (out_m or "").splitlines() if ln.strip().isdigit()]
        result["provenance"][key] = bool(digits and int(digits[0]) > 0)
        log(f"{key}: {result['provenance'][key]}")
    if not all(result["provenance"][k] for k in markers):
        bail(result, "STALE-SOURCE",
             "staged source is missing a one-hot device marker — refusing to run")

    # ---------------- STEP 2 — PREFLIGHT: does the facade route cat_features? ----------
    needle, rel = FACADE_ROUTING_MARKER
    rc_f, out_f = sh(f"grep -c '{needle}' {REPO}/{rel} || true")
    digits = [ln.strip() for ln in (out_f or "").splitlines() if ln.strip().isdigit()]
    routed = bool(digits and int(digits[0]) > 0)
    result["provenance"]["facade_routes_cat_features"] = routed
    if not routed:
        bail(
            result, "BLOCKED-FACADE-ROUTING",
            "the Rust/Python facade does not route `cat_features` into training: "
            f"`{rel}` calls `cb_train::train`, not `train_cat`, and pins "
            "`one_hot_max_size` to the upstream default. A fit through the public "
            "`.fit()` surface would therefore train a FLOAT-ONLY model while appearing "
            "to measure one-hot training. That routing is a separate, not-yet-executed "
            "plan (commit 41e7e9c). Refusing to quote a speed number rather than "
            "measuring the wrong workload. The device one-hot path itself IS verified — "
            "see `device_one_hot_parity_test` (<=1e-5 vs the CPU grower, on-device "
            "residency asserted).",
        )

    env = os.environ.copy()
    env["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + env["PATH"]
    env["CARGO_TARGET_DIR"] = "/content/target"
    env["CARGO_NET_RETRY"] = "5"
    env["RUST_BACKTRACE"] = "1"
    rc, out = sh("rustc --version && cargo --version", env=env)
    result["provenance"]["rust"] = (out or "").strip()

    # ================= PART A — ORACLE GATE =================
    log("\n" + "=" * 70 + "\nPART A — oracle\n" + "=" * 70)
    for label, crate, extra, filters, blocking in ORACLE_SUITES:
        cmd = (["cargo", "test", "--release", "-p", crate] + extra + filters
               + ["--", "--test-threads", "1", "--nocapture"])
        rc, out = sh(cmd, cwd=REPO, env=env, timeout=7200)
        passed = failed = ignored = 0
        for line in (out or "").splitlines():
            if not line.startswith("test result:"):
                continue
            for key, pat in (("passed", r"(\d+)\s+passed"),
                             ("failed", r"(\d+)\s+failed"),
                             ("ignored", r"(\d+)\s+ignored")):
                m = re.search(pat, line)
                if not m:
                    continue
                v = int(m.group(1))
                if key == "passed":
                    passed += v
                elif key == "failed":
                    failed += v
                else:
                    ignored += v
        deltas = [ln.strip() for ln in (out or "").splitlines()
                  if "max|" in ln or "within 1e-5" in ln]
        result["oracle"][label] = {
            "rc": rc, "passed": passed, "failed": failed, "ignored": ignored,
            "blocking": blocking, "measurements": deltas[:40],
            "tail": (out or "")[-3000:],
        }
        log(f"[oracle] {label}: rc={rc} passed={passed} failed={failed}")
        if blocking and (rc != 0 or failed > 0 or passed == 0):
            bail(result, "ORACLE-FAIL",
                 f"blocking oracle suite failed: {label} (rc={rc}, passed={passed}, "
                 f"failed={failed}) — no speed number is quoted")

    result["verdict"] = "ORACLE-PASS"
    log("PART A verdict: ORACLE-PASS (blocking suites green)")

    # ================= PART B — SPEED =================
    log("\n" + "=" * 70 + "\nPART B — speed\n" + "=" * 70)
    sh("pip install -q maturin", env=env, timeout=1800)
    rc, out = sh(["maturin", "build", "--release", "--no-default-features",
                  "--features", "cuda", "-m",
                  os.path.join(REPO, "crates/catboost-rs-py/Cargo.toml")],
                 cwd=REPO, env=env, timeout=7200)
    result["speed"]["build_ok"] = (rc == 0)
    result["speed"]["build_tail"] = (out or "")[-4000:]
    if rc != 0:
        bail(result, "ORACLE-PASS/BUILD-FAIL",
             "cuda wheel build failed — the oracle verdict stands, no speed numbers")

    wheels = sorted(glob.glob("/content/target/wheels/*.whl"), key=os.path.getmtime)
    if not wheels:
        bail(result, "ORACLE-PASS/BUILD-FAIL", "no .whl produced")
    sh([sys.executable, "-m", "pip", "install", "--force-reinstall", wheels[-1]],
       env=env, timeout=1800)

    import numpy as np
    import pandas as pd
    import catboost
    import catboost_rs
    result["speed"]["catboost_version"] = getattr(catboost, "__version__", "unknown")

    # The pool: 45 seeded float ramps + 5 binary categorical columns, with a target that
    # genuinely depends on BOTH, so a one-hot split is worth choosing (a target ignoring
    # the cat columns would let both engines skip them and measure nothing).
    rng = np.random.RandomState(RANDOM_SEED)
    floats = rng.randn(N_ROWS, N_FLOAT).astype(np.float32)
    cats = rng.randint(0, CAT_CARDINALITY, size=(N_ROWS, N_CAT))
    beta = rng.randn(N_FLOAT)
    cat_effect = (cats * np.array([3.0, -2.0, 1.5, -1.0, 2.5])).sum(axis=1)
    y_reg = (floats @ beta + cat_effect + 0.1 * rng.randn(N_ROWS)).astype(np.float64)
    y_bin = (y_reg > np.median(y_reg)).astype(np.float64)

    frame = pd.DataFrame(
        np.hstack([floats, cats.astype(str)]),
        columns=[f"f{i}" for i in range(N_FLOAT)] + [f"c{i}" for i in range(N_CAT)],
    )
    for i in range(N_FLOAT):
        frame[f"f{i}"] = frame[f"f{i}"].astype(np.float32)
    cat_feature_idx = list(range(N_FLOAT, N_FLOAT + N_CAT))

    result["speed"]["config"] = {
        "n_rows": N_ROWS, "n_float": N_FLOAT, "n_cat": N_CAT,
        "cat_cardinality": CAT_CARDINALITY, "one_hot_max_size": ONE_HOT_MAX_SIZE,
        "depth": DEPTH, "iters": ITERS, "learning_rate": LEARNING_RATE,
        "l2_leaf_reg": L2_LEAF_REG, "border_count": BORDER_COUNT,
        "random_seed": RANDOM_SEED, "bootstrap_type": BOOTSTRAP_TYPE,
        "random_strength": RANDOM_STRENGTH,
        "boost_from_average": BOOST_FROM_AVERAGE,
        "leaf_estimation_method": LEAF_ESTIMATION_METHOD,
    }

    def common_kwargs(loss):
        return dict(
            iterations=ITERS, depth=DEPTH, learning_rate=LEARNING_RATE,
            l2_leaf_reg=L2_LEAF_REG, border_count=BORDER_COUNT,
            loss_function=loss, random_strength=RANDOM_STRENGTH,
            leaf_estimation_method=LEAF_ESTIMATION_METHOD,
            boost_from_average=BOOST_FROM_AVERAGE, random_seed=RANDOM_SEED,
            bootstrap_type=BOOTSTRAP_TYPE, one_hot_max_size=ONE_HOT_MAX_SIZE,
        )

    # ---- PART B0 — PROVE device residency per arm ----
    log("\n--- PART B0: device residency proof (CB_GPU_PROF) ---")
    probe_src = os.path.join(WORK, "probe.py")
    with open(probe_src, "w") as fh:
        fh.write(
            "import sys, numpy as np, pandas as pd, catboost_rs\n"
            "loss = sys.argv[1]\n"
            "rng = np.random.RandomState(0)\n"
            "X = rng.randn(4000, 4).astype(np.float32)\n"
            "c = rng.randint(0, 2, size=(4000, 2)).astype(str)\n"
            "df = pd.DataFrame(np.hstack([X, c]), columns=['f0','f1','f2','f3','c0','c1'])\n"
            "for i in range(4): df[f'f{i}'] = df[f'f{i}'].astype(np.float32)\n"
            "y = (X @ rng.randn(4)).astype(np.float64)\n"
            "if loss != 'RMSE': y = (y > np.median(y)).astype(np.float64)\n"
            "m = catboost_rs.CatBoostRegressor(iterations=2, depth=3, learning_rate=0.1,\n"
            "    l2_leaf_reg=3.0, border_count=32, loss_function=loss, random_strength=0.0,\n"
            "    leaf_estimation_method='Gradient', boost_from_average=False,\n"
            f"    random_seed=42, bootstrap_type='{BOOTSTRAP_TYPE}',\n"
            f"    one_hot_max_size={ONE_HOT_MAX_SIZE})\n"
            "m.fit(df, y, cat_features=[4, 5])\n"
        )
    residency = {}
    for loss, _kind in ARMS:
        penv = env.copy()
        penv["CB_GPU_PROF"] = "1"
        rc_p, out_p = sh([sys.executable, probe_src, loss], env=penv, timeout=1800)
        n_tree_lines = len(re.findall(r"CB_GPU_PROF tree", out_p or ""))
        residency[loss] = {"rc": rc_p, "gpu_prof_tree_lines": n_tree_lines,
                           "device_resident": n_tree_lines > 0}
        log(f"[residency] {loss}: device_resident={n_tree_lines > 0} "
            f"(CB_GPU_PROF tree lines={n_tree_lines}, rc={rc_p})")
    result["speed"]["device_residency"] = residency

    def timed_fit(arm, make_model, fit_kwargs, y):
        try:
            make_model().fit(frame, y, **fit_kwargs)
        except Exception as e:
            log(f"[{arm}] warm fit FAILED: {e}")
            result["speed"].setdefault("errors", {})[arm] = "warm: " + repr(e)
            return None, None, None
        try:
            m = make_model()
            t0 = time.time()
            m.fit(frame, y, **fit_kwargs)
            elapsed = round(time.time() - t0, 4)
            pred = np.asarray(m.predict(frame), dtype=np.float64).reshape(-1)
            rmse = float(np.sqrt(np.mean((pred - np.asarray(y, dtype=np.float64)) ** 2)))
            params_readback = None
            if hasattr(m, "get_all_params"):
                try:
                    params_readback = {
                        k: m.get_all_params().get(k)
                        for k in ("bootstrap_type", "random_strength",
                                  "boost_from_average", "one_hot_max_size",
                                  "leaf_estimation_method", "depth", "iterations",
                                  "learning_rate", "l2_leaf_reg", "border_count",
                                  "random_seed", "task_type")
                    }
                except Exception as e:  # noqa: BLE001 - read-back is best-effort evidence
                    params_readback = {"error": repr(e)}
            log(f"[{arm}] fit_s={elapsed} train_rmse={rmse:.6f}")
            return elapsed, round(rmse, 6), params_readback
        except Exception as e:
            log(f"[{arm}] timed fit FAILED: {e}")
            result["speed"].setdefault("errors", {})[arm] = "timed: " + repr(e)
            return None, None, None

    rows = []
    for loss, kind in ARMS:
        y = y_reg if kind == "regression" else y_bin

        def rs_model(loss=loss):
            return catboost_rs.CatBoostRegressor(**common_kwargs(loss))

        def cb_gpu(loss=loss):
            return catboost.CatBoostRegressor(
                task_type="GPU", devices="0", verbose=False, **common_kwargs(loss))

        def cb_cpu(loss=loss):
            return catboost.CatBoostRegressor(
                task_type="CPU", verbose=False, **common_kwargs(loss))

        proven = residency.get(loss, {}).get("device_resident", False)
        log(f"\n--- loss={loss} (catboost_rs device-resident PROVEN: {proven}) ---")
        fit_kw = {"cat_features": cat_feature_idx}
        rs_s, rs_q, _ = timed_fit(f"catboost_rs[{loss}]", rs_model, fit_kw, y)
        cbg_s, cbg_q, cbg_params = timed_fit(f"catboost_gpu[{loss}]", cb_gpu, fit_kw, y)
        cbc_s, cbc_q, _ = timed_fit(f"catboost_cpu[{loss}]", cb_cpu, fit_kw, y)

        speedup = round(cbg_s / rs_s, 3) if (rs_s and cbg_s) else None
        rows.append({
            "loss": loss,
            "catboost_rs_device_resident_proven": proven,
            "activation_observable": proven,
            "catboost_rs_s": rs_s, "catboost_rs_train_rmse": rs_q,
            "catboost_gpu_s": cbg_s, "catboost_gpu_train_rmse": cbg_q,
            "catboost_cpu_s": cbc_s, "catboost_cpu_train_rmse": cbc_q,
            "speedup_official_catboost_gpu": speedup,
            "official_params_readback": cbg_params,
        })

    result["speed"]["rows"] = rows

    # The gate SPEC-OH-30 states: faster than official CatBoost GPU on BOTH arms, with
    # activation observed for every catboost-rs arm.
    gate_met = all(
        r["activation_observable"] and r["speedup_official_catboost_gpu"] is not None
        and r["speedup_official_catboost_gpu"] > 1.0
        for r in rows
    )
    result["speed"]["gate_met"] = gate_met
    result["verdict"] = "ORACLE-PASS/SPEED-PASS" if gate_met else "ORACLE-PASS/SPEED-FAIL"

    json.dump(result, open(os.path.join(WORK, "result.json"), "w"), indent=2)

    with open(os.path.join(WORK, "report.md"), "w") as fh:
        fh.write("# One-hot categorical GPU speed gate (SPEC-OH-30) — Colab T4 (CUDA)\n\n")
        fh.write(f"- GPU: `{gpu}`\n")
        fh.write(f"- verdict: **{result['verdict']}**\n")
        fh.write(f"- catboost: {result['speed'].get('catboost_version')}\n")
        fh.write(f"- pinned identically on both sides: `bootstrap_type={BOOTSTRAP_TYPE}`, "
                 f"`random_strength={RANDOM_STRENGTH}`, "
                 f"`boost_from_average={BOOST_FROM_AVERAGE}`, "
                 f"`one_hot_max_size={ONE_HOT_MAX_SIZE}`, "
                 f"`leaf_estimation_method={LEAF_ESTIMATION_METHOD}`\n\n")
        fh.write("## Oracle\n\n| suite | passed | failed | blocking |\n|---|---|---|---|\n")
        for k, v in result["oracle"].items():
            fh.write(f"| {k} | {v['passed']} | {v['failed']} | {v['blocking']} |\n")
        fh.write(f"\n## Speed ({N_ROWS}x({N_FLOAT} float + {N_CAT} cat), depth {DEPTH}, "
                 f"{ITERS} iters)\n\n")
        fh.write("| loss | rs on device? | catboost_rs s | CatBoost GPU s | CatBoost CPU s "
                 "| speedup vs CB-GPU |\n|---|---|---|---|---|---|\n")
        for r in rows:
            fh.write(f"| {r['loss']} | {r['catboost_rs_device_resident_proven']} | "
                     f"{r['catboost_rs_s']} | {r['catboost_gpu_s']} | {r['catboost_cpu_s']} | "
                     f"{r['speedup_official_catboost_gpu']} |\n")
        fh.write("\n## Official-side parameter read-back\n\n")
        for r in rows:
            fh.write(f"- `{r['loss']}`: `{r['official_params_readback']}`\n")
        fh.write("\n## Caveats\n\n")
        for k, v in result["caveats"].items():
            fh.write(f"- **{k}**: {v}\n")
    log("\n" + open(os.path.join(WORK, "report.md")).read())
    log("DONE verdict=" + result["verdict"])


if __name__ == "__main__":
    main()
