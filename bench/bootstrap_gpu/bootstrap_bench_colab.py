#!/usr/bin/env python3
# catboost-rs WR-01 bootstrap oracle + learning-speed sweep — Google Colab T4 (CUDA).
#
# The Colab sibling of `bootstrap_bench.py` (Kaggle P100). Same two questions, same
# honesty discipline, three deliberate differences:
#
#   1. Source is the tree ALREADY STAGED at /content/cbrs (uploaded by the driver),
#      not a git clone — so a run measures the working tree under test, including
#      uncommitted WR-01 changes. Provenance markers below prove WHICH tree it is.
#   2. The `only_No_is_gpu_eligible` caveat is GONE, because it is no longer true:
#      WR-01 made Bayesian / Bernoulli / MVS device-eligible for the oblivious grow.
#      Poisson remains rejected on every backend by design.
#   3. The `device_activation_not_observable` caveat is CLOSED, not restated. Part B0
#      runs one short fit per arm under `CB_GPU_PROF=1` and greps the per-tree device
#      profiling lines the resident grower emits. An arm that prints no `CB_GPU_PROF
#      tree` line did NOT run on the device, and its speed row is labelled accordingly
#      instead of being quietly presented as a GPU number.
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

SPEED_CONFIG = dict(n_rows=300_000, n_features=50, seed=42)
DEPTH = 6
ITERS = 30
LEARNING_RATE = 0.1
L2_LEAF_REG = 3.0
BORDER_COUNT = 32
RANDOM_SEED = 42

# `subsample` / `bagging_temperature` mirror the oracle fixtures' pinning so each arm
# exercises a real sampler rather than a degenerate all-1.0 short-circuit. Poisson is
# included to RECORD the uniform rejection, not as a pass condition.
BOOTSTRAP_ARMS = [
    ("No", {}),
    ("Bayesian", {"bagging_temperature": 1.0}),
    ("Bernoulli", {"subsample": 0.8}),
    ("MVS", {"subsample": 0.8}),
    ("Poisson", {"subsample": 0.8}),
]
# Which arms WR-01 makes device-eligible for the oblivious grow.
GPU_ELIGIBLE = {"No", "Bayesian", "Bernoulli", "MVS"}

# (label, crate, cargo extra args, test filters, blocking?)
ORACLE_SUITES = [
    ("cb-train bootstrap parity (CPU, frozen bias!=0 family)", "cb-train",
     ["--test", "bootstrap_oracle_test"], [], True),
    ("cb-train bias-0 upstream oracle (CUDA device vs upstream)", "cb-train",
     ["--no-default-features", "--features", "cuda", "--test",
      "bootstrap_dev_oracle_test"], [], True),
    ("cb-train WR-01 device bootstrap parity (CUDA)", "cb-train",
     ["--no-default-features", "--features", "cuda", "--test",
      "device_bootstrap_parity_test"], [], True),
    ("cb-train device draw replay (host RNG phase)", "cb-train",
     ["--lib"], ["device_draw_replay"], True),
    ("cb-backend device bootstrap kernels (CUDA)", "cb-backend",
     ["--no-default-features", "--features", "cuda", "--lib"], ["bootstrap"], False),
    ("cb-backend device MVS kernels (CUDA)", "cb-backend",
     ["--no-default-features", "--features", "cuda", "--lib"], ["mvs"], False),
]


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

    result = {"provenance": {}, "oracle": {}, "speed": {}, "caveats": {}}
    result["caveats"]["gpu_eligibility"] = (
        "WR-01 made bootstrap_type in {No, Bayesian, Bernoulli, MVS} device-eligible "
        "for the SymmetricTree (oblivious) grow via host sampling (Design A). Poisson "
        "is rejected up front on EVERY backend by design. Non-oblivious grow policies "
        "x sampling remain CPU-only."
    )
    result["caveats"]["device_activation_is_proven_per_arm"] = (
        "Part B0 proves device residency per arm by running a short fit under "
        "CB_GPU_PROF=1 and requiring the per-tree 'CB_GPU_PROF tree' line the resident "
        "grower emits. Arms without it are reported as CPU rows."
    )
    result["caveats"]["mvs_upstream_tree2"] = (
        "MVS is gated against upstream over trees 0-1 only (bootstrap_dev_oracle_test "
        "MVS_GATED_TREES=2): a PRE-EXISTING CPU-side MVS tree-2 sampling gap, unrelated "
        "to the device work. Device-vs-CPU MVS parity is locked at <=1e-5 separately."
    )

    # ---------------- STEP 1 — provenance ----------------
    rc, out = sh("nvidia-smi --query-gpu=name,driver_version,memory.total "
                 "--format=csv,noheader")
    result["provenance"]["gpu"] = (out or "").strip().splitlines()[0] if rc == 0 and out.strip() else None
    if not result["provenance"]["gpu"]:
        result["fatal"] = "no GPU visible to nvidia-smi — refusing to quote GPU numbers"
        json.dump(result, open(os.path.join(WORK, "result.json"), "w"), indent=2)
        log("FATAL:", result["fatal"])
        sys.exit(2)
    log("GPU:", result["provenance"]["gpu"])
    result["provenance"]["cpu_count"] = os.cpu_count()
    result["provenance"]["source"] = REPO

    # Provenance markers: prove the staged tree really carries the WR-01 wiring, so a
    # stale upload can never masquerade as a verified run.
    markers = {
        "has_rng_fix": ("POST_TREE_EXTRA_DRAWS: usize = 2",
                        "crates/cb-train/src/boosting.rs"),
        "has_lds_part_update": ("partition_update_lds_kernel",
                                "crates/cb-backend/src/kernels.rs"),
        "has_sample_from_host": ("sample_from_host",
                                 "crates/cb-compute/src/runtime.rs"),
        "has_score_channel_split": ("score_der1_h",
                                    "crates/cb-backend/src/gpu_runtime/mod.rs"),
        "has_draw_replay": ("replay_grow_draws",
                            "crates/cb-train/src/device_draw_replay.rs"),
    }
    for key, (needle, rel) in markers.items():
        rc_m, out_m = sh(f"grep -c '{needle}' {REPO}/{rel} || true")
        digits = [ln.strip() for ln in (out_m or "").splitlines() if ln.strip().isdigit()]
        result["provenance"][key] = bool(digits and int(digits[0]) > 0)
        log(f"{key}: {result['provenance'][key]}")
    if not all(result["provenance"][k] for k in markers):
        result["fatal"] = "staged source is missing a WR-01 marker — refusing to run"
        json.dump(result, open(os.path.join(WORK, "result.json"), "w"), indent=2)
        log("FATAL:", result["fatal"])
        sys.exit(2)

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
               + ["--", "--include-ignored", "--test-threads", "1", "--nocapture"])
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
        # Keep the measured parity numbers the device suites print, as evidence.
        deltas = [ln.strip() for ln in (out or "").splitlines()
                  if "max|" in ln or "within 1e-5" in ln]
        result["oracle"][label] = {
            "rc": rc, "passed": passed, "failed": failed, "ignored": ignored,
            "blocking": blocking, "measurements": deltas[:40],
            "tail": (out or "")[-3000:],
        }
        log(f"[oracle] {label}: rc={rc} passed={passed} failed={failed}")
        if blocking and (rc != 0 or failed > 0 or passed == 0):
            result["verdict"] = "ORACLE-FAIL"
            result["fatal"] = (f"blocking oracle suite failed: {label} "
                               f"(rc={rc}, passed={passed}, failed={failed}) — "
                               "no speed number is quoted")
            json.dump(result, open(os.path.join(WORK, "result.json"), "w"), indent=2)
            log("FATAL:", result["fatal"])
            sys.exit(2)

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
        result["verdict"] = "ORACLE-PASS/BUILD-FAIL"
        json.dump(result, open(os.path.join(WORK, "result.json"), "w"), indent=2)
        log("FATAL: cuda wheel build failed — oracle stands, no speed numbers")
        sys.exit(2)

    wheels = sorted(glob.glob("/content/target/wheels/*.whl"), key=os.path.getmtime)
    if not wheels:
        result["verdict"] = "ORACLE-PASS/BUILD-FAIL"
        result["fatal"] = "no .whl produced"
        json.dump(result, open(os.path.join(WORK, "result.json"), "w"), indent=2)
        sys.exit(2)
    sh([sys.executable, "-m", "pip", "install", "--force-reinstall", wheels[-1]],
       env=env, timeout=1800)

    import numpy as np
    sys.path.insert(0, os.path.join(REPO, "bench"))
    import generator
    import catboost_rs
    import catboost
    result["speed"]["catboost_version"] = getattr(catboost, "__version__", "unknown")

    X, y = generator.generate(**SPEED_CONFIG)
    result["speed"]["config"] = {
        "speed_config": SPEED_CONFIG, "depth": DEPTH, "iters": ITERS,
        "learning_rate": LEARNING_RATE, "l2_leaf_reg": L2_LEAF_REG,
        "border_count": BORDER_COUNT, "random_seed": RANDOM_SEED,
        "loss": "RMSE", "X_shape": list(X.shape),
    }

    # ---- PART B0 — PROVE device residency per arm (closes the old caveat) ----
    # A tiny out-of-process fit per arm with CB_GPU_PROF=1. The resident oblivious
    # grower prints one "CB_GPU_PROF tree ..." line per tree; a CPU-fallback fit
    # prints none. This is the ONLY thing that makes the speed rows below honest.
    log("\n--- PART B0: device residency proof (CB_GPU_PROF) ---")
    probe_src = os.path.join(WORK, "probe.py")
    with open(probe_src, "w") as fh:
        fh.write(
            "import sys, json, numpy as np, catboost_rs\n"
            "bt = sys.argv[1]; extra = json.loads(sys.argv[2])\n"
            "rng = np.random.RandomState(0)\n"
            "X = rng.randn(4000, 8).astype(np.float32)\n"
            "y = (X @ rng.randn(8)).astype(np.float32)\n"
            "m = catboost_rs.CatBoostRegressor(iterations=2, depth=3, learning_rate=0.1,\n"
            "    l2_leaf_reg=3.0, border_count=32, loss_function='RMSE', random_strength=0.0,\n"
            "    leaf_estimation_method='Gradient', boost_from_average=False,\n"
            "    random_seed=42, bootstrap_type=bt, **extra)\n"
            "m.fit(X, y)\n"
        )
    residency = {}
    for bt, extra in BOOTSTRAP_ARMS:
        penv = env.copy()
        penv["CB_GPU_PROF"] = "1"
        rc_p, out_p = sh([sys.executable, probe_src, bt, json.dumps(extra)],
                         env=penv, timeout=1800)
        n_tree_lines = len(re.findall(r"CB_GPU_PROF tree", out_p or ""))
        residency[bt] = {"rc": rc_p, "gpu_prof_tree_lines": n_tree_lines,
                         "device_resident": n_tree_lines > 0}
        log(f"[residency] {bt}: device_resident={n_tree_lines > 0} "
            f"(CB_GPU_PROF tree lines={n_tree_lines}, rc={rc_p})")
    result["speed"]["device_residency"] = residency

    def timed_fit(arm, make_model):
        try:
            make_model().fit(X, y)
        except Exception as e:
            log(f"[{arm}] warm fit FAILED: {e}")
            result["speed"].setdefault("errors", {})[arm] = "warm: " + repr(e)
            return None, None
        try:
            m = make_model()
            t0 = time.time()
            m.fit(X, y)
            pred = np.asarray(m.predict(X), dtype=np.float64).reshape(-1)
            elapsed = round(time.time() - t0, 4)
            rmse = float(np.sqrt(np.mean((pred - np.asarray(y, dtype=np.float64)) ** 2)))
            log(f"[{arm}] fit_s={elapsed} train_rmse={rmse:.6f}")
            return elapsed, round(rmse, 6)
        except Exception as e:
            log(f"[{arm}] timed fit FAILED: {e}")
            result["speed"].setdefault("errors", {})[arm] = "timed: " + repr(e)
            return None, None

    rows = []
    for bt, extra in BOOTSTRAP_ARMS:
        def rs_model(bt=bt, extra=extra):
            return catboost_rs.CatBoostRegressor(
                iterations=ITERS, depth=DEPTH, learning_rate=LEARNING_RATE,
                l2_leaf_reg=L2_LEAF_REG, border_count=BORDER_COUNT,
                loss_function="RMSE", random_strength=0.0,
                leaf_estimation_method="Gradient", boost_from_average=False,
                random_seed=RANDOM_SEED, bootstrap_type=bt, **extra)

        def cb_gpu(bt=bt, extra=extra):
            return catboost.CatBoostRegressor(
                iterations=ITERS, depth=DEPTH, learning_rate=LEARNING_RATE,
                l2_leaf_reg=L2_LEAF_REG, border_count=BORDER_COUNT,
                loss_function="RMSE", random_strength=0.0,
                leaf_estimation_method="Gradient", boost_from_average=False,
                random_seed=RANDOM_SEED, bootstrap_type=bt,
                task_type="GPU", devices="0", verbose=False, **extra)

        def cb_cpu(bt=bt, extra=extra):
            return catboost.CatBoostRegressor(
                iterations=ITERS, depth=DEPTH, learning_rate=LEARNING_RATE,
                l2_leaf_reg=L2_LEAF_REG, border_count=BORDER_COUNT,
                loss_function="RMSE", random_strength=0.0,
                leaf_estimation_method="Gradient", boost_from_average=False,
                random_seed=RANDOM_SEED, bootstrap_type=bt,
                task_type="CPU", verbose=False, **extra)

        proven = residency.get(bt, {}).get("device_resident", False)
        log(f"\n--- bootstrap_type={bt} (catboost_rs device-resident PROVEN: {proven}) ---")
        rs_s, rs_q = timed_fit(f"catboost_rs[{bt}]", rs_model)
        cbg_s, cbg_q = timed_fit(f"catboost_gpu[{bt}]", cb_gpu)
        cbc_s, cbc_q = timed_fit(f"catboost_cpu[{bt}]", cb_cpu)

        ratio = None
        if rs_s and cbg_s:
            ratio = round(rs_s / cbg_s, 3)
        rows.append({
            "bootstrap_type": bt,
            "expected_gpu_eligible": bt in GPU_ELIGIBLE,
            "catboost_rs_device_resident_proven": proven,
            "catboost_rs_s": rs_s, "catboost_rs_train_rmse": rs_q,
            "catboost_gpu_s": cbg_s, "catboost_gpu_train_rmse": cbg_q,
            "catboost_cpu_s": cbc_s, "catboost_cpu_train_rmse": cbc_q,
            "ratio_rs_over_catboost_gpu": ratio,
        })

    result["speed"]["rows"] = rows

    # A single, checkable consistency claim: every arm WR-01 says is eligible must
    # have been PROVEN device-resident. A mismatch is a silent-fallback regression and
    # is recorded as such rather than left for a reader to notice.
    mismatches = [r["bootstrap_type"] for r in rows
                  if r["expected_gpu_eligible"] and not r["catboost_rs_device_resident_proven"]]
    result["speed"]["eligibility_mismatches"] = mismatches
    if mismatches:
        result["verdict"] = "ORACLE-PASS/RESIDENCY-MISMATCH"
        log(f"WARNING: arms expected on device but not proven resident: {mismatches}")

    json.dump(result, open(os.path.join(WORK, "result.json"), "w"), indent=2)

    # ---- report ----
    with open(os.path.join(WORK, "report.md"), "w") as fh:
        fh.write("# WR-01 bootstrap oracle + speed — Colab T4 (CUDA)\n\n")
        fh.write(f"- GPU: `{result['provenance']['gpu']}`\n")
        fh.write(f"- verdict: **{result['verdict']}**\n")
        fh.write(f"- catboost: {result['speed'].get('catboost_version')}\n\n")
        fh.write("## Oracle\n\n| suite | passed | failed | blocking |\n|---|---|---|---|\n")
        for k, v in result["oracle"].items():
            fh.write(f"| {k} | {v['passed']} | {v['failed']} | {v['blocking']} |\n")
        fh.write(f"\n## Speed ({SPEED_CONFIG['n_rows']}x{SPEED_CONFIG['n_features']}, "
                 f"depth {DEPTH}, {ITERS} iters, RMSE)\n\n")
        fh.write("| bootstrap | rs on device? | catboost_rs s | CatBoost GPU s | "
                 "CatBoost CPU s | rs/CB-GPU |\n|---|---|---|---|---|---|\n")
        for r in rows:
            fh.write(f"| {r['bootstrap_type']} | {r['catboost_rs_device_resident_proven']} | "
                     f"{r['catboost_rs_s']} | {r['catboost_gpu_s']} | {r['catboost_cpu_s']} | "
                     f"{r['ratio_rs_over_catboost_gpu']} |\n")
        fh.write("\n## Caveats\n\n")
        for k, v in result["caveats"].items():
            fh.write(f"- **{k}**: {v}\n")
    log("\n" + open(os.path.join(WORK, "report.md")).read())
    log("DONE verdict=" + result["verdict"])


if __name__ == "__main__":
    main()
