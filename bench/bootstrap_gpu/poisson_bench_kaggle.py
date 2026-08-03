#!/usr/bin/env python3
# catboost-rs GPU-only Poisson bootstrap — oracle gate + learning-speed sweep.
# Kaggle notebook, accelerator pinned to NvidiaTeslaP100 (kernel-metadata machine_shape).
#
# WHAT THIS ANSWERS, in order, and why the order is not negotiable:
#
#   Part A (BLOCKING) — does the device Poisson kernel still reproduce upstream
#     BIT-FOR-BIT on CUDA? `bootstrap_type=Poisson` is upstream CatBoost's GPU-ONLY
#     sampler: its CPU validator rejects it outright ("poisson bootstrap is not
#     supported on CPU", bootstrap_options.cpp:29), and the per-object weights its CUDA
#     kernel draws are not observable through any public API. So there is no
#     CatBoost-Python reference to diff against, on either task type. The reference is
#     instead cb-oracle/generator/poisson_bootstrap_oracle.cpp — a verbatim HOST
#     transcription of upstream's PoissonBootstrapImpl + random_gen.cuh — frozen into
#     crates/cb-oracle/fixtures/bootstrap_poisson/. Part A re-runs that gate on the GPU
#     under test. If it fails, NO speed number is quoted.
#
#   Part B — how fast is it, against official CatBoost GPU on the SAME machine?
#     This is the only leg that needs CUDA specifically: CatBoost's GPU trainer is
#     CUDA-only, so `task_type='GPU', bootstrap_type='Poisson'` cannot be run anywhere
#     else. Note what this comparison is and is not: it is a SPEED and model-quality
#     comparison, NOT a numeric parity gate. Upstream's GPU trainer differs from this
#     implementation in quantization and histogram details for every bootstrap type, so
#     predictions are not expected to agree to 1e-5 and are not asserted to.
#
# HONESTY DISCIPLINE (bench/RESULTS.md house style):
#   * Correctness is gated BEFORE any timing.
#   * Every number comes from a measured call in THIS run; a failed arm leaves `null`,
#     never an invented value.
#   * Device residency is PROVEN per arm (Part B0, CB_GPU_PROF) rather than assumed. An
#     arm that prints no per-tree device line is labelled a CPU row, not quietly
#     presented as a GPU number.
#   * The GPU model is read from nvidia-smi and recorded. If it is not a P100 the run
#     still completes and reports what it actually got — the request was P100, the
#     record is whatever Kaggle assigned.
#
# Source arrives as a private Kaggle DATASET (a tarball of the working tree), not a git
# clone: the tree under test has uncommitted work, and a dataset guarantees the kernel
# runs exactly the bytes that were tested locally. Provenance markers below prove the
# extracted tree really carries the Poisson wiring, so a stale upload cannot masquerade
# as a verified run.

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time

WORK = "/kaggle/working"
REPO = "/kaggle/tmp/cbrs"
# Kaggle mounts datasets at /kaggle/input/datasets/<owner>/<slug>/, NOT /kaggle/input/<slug>/
# (verified by probe kernel boomvector/cbrs-input-probe, 2026-07-31). Matched recursively so
# either layout works.
SRC_GLOB = "/kaggle/input/**/catboost_rs_src.tar.gz"

SPEED_CONFIG = dict(n_rows=300_000, n_features=50, seed=42)
DEPTH = 6
ITERS = 30
LEARNING_RATE = 0.1
L2_LEAF_REG = 3.0
BORDER_COUNT = 32
RANDOM_SEED = 42

# Poisson first — it is the subject. The other arms are the control group: they show
# whether a Poisson-specific number is really Poisson-specific or just this machine.
#
# `subsample` must be < 1 for Poisson: upstream's
# GetPoissonLambda() = -log(1 - subsample) returns -1 at subsample >= 1, which zeroes
# every sample weight. catboost_rs rejects that configuration instead of training on it.
BOOTSTRAP_ARMS = [
    ("Poisson", {"subsample": 0.8}),
    ("No", {}),
    ("Bernoulli", {"subsample": 0.8}),
    ("Bayesian", {"bagging_temperature": 1.0}),
    ("MVS", {"subsample": 0.8}),
]
GPU_ELIGIBLE = {"No", "Bayesian", "Bernoulli", "MVS", "Poisson"}

# (label, crate, cargo extra args, test filters, blocking?)
ORACLE_SUITES = [
    # THE Poisson gate. Bit-for-bit vs the upstream-transcription fixtures over three
    # launch geometries x two consecutive draws. Blocking: it is the only parity
    # evidence Poisson can have.
    ("Poisson upstream oracle (CUDA, bit-for-bit)", "cb-backend",
     ["--no-default-features", "--features", "cuda", "--lib"],
     ["poisson_bootstrap_oracle_test"], True),
    ("Poisson device e2e (CUDA)", "cb-train",
     ["--no-default-features", "--features", "cuda", "--test",
      "device_poisson_bootstrap_test"], [], True),
    # Regression control: the other three arms must still hold <=1e-5 against upstream
    # CatBoost 1.2.10, since the Poisson work refactored the shared session sampler.
    ("bias-0 device vs UPSTREAM CatBoost (CUDA)", "cb-train",
     ["--no-default-features", "--features", "cuda", "--test",
      "bootstrap_dev_oracle_test"], [], True),
    ("WR-01 device bootstrap parity (CUDA)", "cb-train",
     ["--no-default-features", "--features", "cuda", "--test",
      "device_bootstrap_parity_test"], [], True),
    ("device bootstrap kernels (CUDA)", "cb-backend",
     ["--no-default-features", "--features", "cuda", "--lib"], ["bootstrap"], False),
    ("Poisson parallel-draw speed (CUDA)", "cb-backend",
     ["--no-default-features", "--features", "cuda", "--release", "--lib"],
     ["poisson_bootstrap_speed_test"], False),
    ("sampled fits run at device speed (CUDA)", "cb-train",
     ["--no-default-features", "--features", "cuda", "--release", "--test",
      "device_bootstrap_speed_test"], [], False),
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
    result["caveats"]["poisson_parity_basis"] = (
        "Poisson cannot be gated against a CatBoost-Python run: upstream rejects it on "
        "task_type=CPU, and its per-object GPU bootstrap weights are not exposed by any "
        "public API. Its parity evidence is bit-for-bit agreement between the device "
        "kernel and a verbatim host transcription of upstream's PoissonBootstrapImpl + "
        "random_gen.cuh (three launch geometries, two consecutive draws). The CatBoost "
        "GPU columns below are a SPEED and quality comparison, not a numeric parity gate."
    )
    result["caveats"]["backend_asymmetry"] = (
        "catboost_rs mirrors upstream's asymmetry: Poisson trains on the device and is "
        "REFUSED by the CPU grower, exactly as upstream accepts it on task_type=GPU and "
        "rejects it on task_type=CPU. The CatBoost CPU column is therefore expected to "
        "be null for the Poisson row — that is the correct result, not a failure."
    )
    result["caveats"]["device_residency_proven"] = (
        "Part B0 proves device residency per arm with CB_GPU_PROF=1 and the per-tree "
        "device lines the resident grower emits. Arms without them are reported as CPU."
    )

    # ---------------- STEP 0 — GPU + source ----------------
    rc, out = sh("nvidia-smi --query-gpu=name,driver_version,memory.total "
                 "--format=csv,noheader")
    gpu = (out or "").strip().splitlines()[0] if rc == 0 and out.strip() else None
    result["provenance"]["gpu"] = gpu
    if not gpu:
        result["fatal"] = "no GPU visible to nvidia-smi — refusing to quote GPU numbers"
        json.dump(result, open(os.path.join(WORK, "result.json"), "w"), indent=2)
        log("FATAL:", result["fatal"])
        sys.exit(2)
    result["provenance"]["gpu_is_p100"] = "P100" in gpu
    log("GPU:", gpu, "| P100 requested:", result["provenance"]["gpu_is_p100"])
    result["provenance"]["cpu_count"] = os.cpu_count()

    # Kaggle DECOMPRESSES uploaded archives when it ingests a dataset, so the source may
    # arrive either as the tarball itself or already expanded into the input directory.
    # Handle both rather than assuming: which one you get is Kaggle's choice, not ours.
    # Either way the tree is COPIED to a writable path — /kaggle/input is read-only and
    # cargo must write.
    os.makedirs(os.path.dirname(REPO), exist_ok=True)
    tarballs = sorted(glob.glob(SRC_GLOB, recursive=True))
    if tarballs:
        os.makedirs(REPO, exist_ok=True)
        with tarfile.open(tarballs[0]) as tf:
            tf.extractall(REPO)
        result["provenance"]["source"] = tarballs[0]
        log("extracted", tarballs[0], "->", REPO)
    else:
        # Search for the workspace root at ANY depth rather than assuming the dataset
        # mounts as /kaggle/input/<slug>/crates: Kaggle decides both the mount name and
        # whether an uploaded archive is expanded, and a wrong guess here costs a whole
        # rebuild to discover.
        roots = []
        for base in sorted(glob.glob("/kaggle/input/*")):
            for dirpath, dirnames, filenames in os.walk(base):
                if "crates" in dirnames and "Cargo.toml" in filenames:
                    roots.append(dirpath)
                    dirnames[:] = []
                if dirpath.count("/") > 6:
                    dirnames[:] = []
        if not roots:
            # Dump what IS mounted, so one failed run is enough to diagnose.
            listing = []
            for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
                listing.append(f"{dirpath} dirs={dirnames[:10]} files={filenames[:10]}")
                if len(listing) > 60:
                    break
            result["input_listing"] = listing
            result["fatal"] = (
                f"source not found: no tarball matching {SRC_GLOB} and nothing under "
                "/kaggle/input containing crates/ + Cargo.toml"
            )
            json.dump(result, open(os.path.join(WORK, "result.json"), "w"), indent=2)
            log("FATAL:", result["fatal"])
            log("/kaggle/input listing:\n" + "\n".join(listing))
            sys.exit(2)
        shutil.copytree(roots[0], REPO, dirs_exist_ok=True)
        result["provenance"]["source"] = roots[0]
        log("copied", roots[0], "->", REPO)

    # Provenance markers: a stale upload must not be able to pass as a verified run.
    markers = {
        "has_poisson_kernel": ("poisson_bootstrap_kernel",
                               "crates/cb-backend/src/kernels/bootstrap_device.rs"),
        "has_upstream_next_poisson": ("NextUniform",
                                      "crates/cb-backend/src/kernels/bootstrap_device.rs"),
        "has_poisson_seeds": ("create_poisson_seeds",
                              "crates/cb-backend/src/gpu_runtime/session.rs"),
        "has_poisson_wiring": ("device_poisson", "crates/cb-train/src/boosting.rs"),
        "has_poisson_oracle_test": ("bit_for_bit",
                                    "crates/cb-backend/src/kernels/poisson_bootstrap_oracle_test.rs"),
        "has_poisson_fixtures": ("stride",
                                 "crates/cb-oracle/fixtures/bootstrap_poisson/wide/config.json"),
    }
    for key, (needle, rel) in markers.items():
        rc_m, out_m = sh(f"grep -c '{needle}' {REPO}/{rel} || true")
        digits = [ln.strip() for ln in (out_m or "").splitlines() if ln.strip().isdigit()]
        result["provenance"][key] = bool(digits and int(digits[0]) > 0)
        log(f"{key}: {result['provenance'][key]}")
    if not all(result["provenance"][k] for k in markers):
        result["fatal"] = "extracted source is missing a Poisson marker — refusing to run"
        json.dump(result, open(os.path.join(WORK, "result.json"), "w"), indent=2)
        log("FATAL:", result["fatal"])
        sys.exit(2)

    # ---------------- STEP 1 — toolchain ----------------
    env = os.environ.copy()
    env["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + env["PATH"]
    env["CARGO_TARGET_DIR"] = "/kaggle/tmp/target"
    env["CARGO_NET_RETRY"] = "5"
    env["RUST_BACKTRACE"] = "1"
    rc, _ = sh("rustc --version", env=env)
    if rc != 0:
        sh("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal "
           "--default-toolchain stable", env=env, timeout=3600)
    rc, out = sh("rustc --version && cargo --version && nvcc --version | tail -2",
                 env=env)
    result["provenance"]["toolchain"] = (out or "").strip()

    # ================= PART A — ORACLE GATE =================
    log("\n" + "=" * 70 + "\nPART A — oracle (blocking)\n" + "=" * 70)
    for label, crate, extra, filters, blocking in ORACLE_SUITES:
        cmd = (["cargo", "test", "-p", crate] + extra + filters
               + ["--", "--test-threads", "1", "--nocapture"])
        rc, out = sh(cmd, cwd=REPO, env=env, timeout=10800)
        passed = failed = 0
        for line in (out or "").splitlines():
            if not line.startswith("test result:"):
                continue
            m = re.search(r"(\d+)\s+passed", line)
            if m:
                passed += int(m.group(1))
            m = re.search(r"(\d+)\s+failed", line)
            if m:
                failed += int(m.group(1))
        # Keep the measured evidence the suites print.
        keep = [ln.strip() for ln in (out or "").splitlines()
                if ("bit-for-bit" in ln or "max|" in ln or "[speed]" in ln
                    or "[poisson" in ln or "within 1e-5" in ln)]
        result["oracle"][label] = {
            "rc": rc, "passed": passed, "failed": failed, "blocking": blocking,
            "measurements": keep[:60], "tail": (out or "")[-3000:],
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
    log("PART A verdict: ORACLE-PASS")

    # ================= PART B — SPEED =================
    log("\n" + "=" * 70 + "\nPART B — speed\n" + "=" * 70)
    sh([sys.executable, "-m", "pip", "install", "-q", "maturin"], env=env, timeout=1800)
    rc, out = sh(["maturin", "build", "--release", "--no-default-features",
                  "--features", "cuda", "-m",
                  os.path.join(REPO, "crates/catboost-rs-py/Cargo.toml")],
                 cwd=REPO, env=env, timeout=10800)
    result["speed"]["build_ok"] = (rc == 0)
    result["speed"]["build_tail"] = (out or "")[-4000:]
    wheels = sorted(glob.glob("/kaggle/tmp/target/wheels/*.whl"), key=os.path.getmtime)
    if rc != 0 or not wheels:
        result["verdict"] = "ORACLE-PASS/BUILD-FAIL"
        result["fatal"] = "cuda wheel build failed — the oracle verdict stands, no speed numbers"
        json.dump(result, open(os.path.join(WORK, "result.json"), "w"), indent=2)
        log("FATAL:", result["fatal"])
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

    # ---- PART B0 — PROVE device residency per arm ----
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
            f"(tree lines={n_tree_lines}, rc={rc_p})")
    result["speed"]["device_residency"] = residency

    def timed_fit(arm, make_model):
        try:
            make_model().fit(X, y)
        except Exception as e:
            log(f"[{arm}] warm fit FAILED: {e}")
            result["speed"].setdefault("errors", {})[arm] = "warm: " + repr(e)[:300]
            return None, None
        try:
            m = make_model()
            t0 = time.time()
            m.fit(X, y)
            elapsed = round(time.time() - t0, 4)
            pred = np.asarray(m.predict(X), dtype=np.float64).reshape(-1)
            rmse = float(np.sqrt(np.mean((pred - np.asarray(y, dtype=np.float64)) ** 2)))
            log(f"[{arm}] fit_s={elapsed} train_rmse={rmse:.6f}")
            return elapsed, round(rmse, 6)
        except Exception as e:
            log(f"[{arm}] timed fit FAILED: {e}")
            result["speed"].setdefault("errors", {})[arm] = "timed: " + repr(e)[:300]
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

        rows.append({
            "bootstrap_type": bt,
            "expected_gpu_eligible": bt in GPU_ELIGIBLE,
            "catboost_rs_device_resident_proven": proven,
            "catboost_rs_s": rs_s, "catboost_rs_train_rmse": rs_q,
            "catboost_gpu_s": cbg_s, "catboost_gpu_train_rmse": cbg_q,
            "catboost_cpu_s": cbc_s, "catboost_cpu_train_rmse": cbc_q,
            "ratio_rs_over_catboost_gpu": (round(rs_s / cbg_s, 3)
                                           if rs_s and cbg_s else None),
        })
    result["speed"]["rows"] = rows

    mismatches = [r["bootstrap_type"] for r in rows
                  if r["expected_gpu_eligible"]
                  and not r["catboost_rs_device_resident_proven"]]
    result["speed"]["eligibility_mismatches"] = mismatches
    if mismatches:
        result["verdict"] = "ORACLE-PASS/RESIDENCY-MISMATCH"
        log(f"WARNING: arms expected on device but not proven resident: {mismatches}")

    json.dump(result, open(os.path.join(WORK, "result.json"), "w"), indent=2)

    # ---- report ----
    with open(os.path.join(WORK, "report.md"), "w") as fh:
        fh.write("# GPU-only Poisson bootstrap — oracle + speed (Kaggle CUDA)\n\n")
        fh.write(f"- GPU: `{gpu}` (P100 requested: "
                 f"{result['provenance']['gpu_is_p100']})\n")
        fh.write(f"- verdict: **{result['verdict']}**\n")
        fh.write(f"- catboost: {result['speed'].get('catboost_version')}\n")
        fh.write(f"- toolchain: `{result['provenance'].get('toolchain','')}`\n\n")
        fh.write("## Oracle\n\n| suite | passed | failed | blocking |\n|---|---|---|---|\n")
        for k, v in result["oracle"].items():
            fh.write(f"| {k} | {v['passed']} | {v['failed']} | {v['blocking']} |\n")
        fh.write("\n### Measured evidence\n\n```\n")
        for k, v in result["oracle"].items():
            for m in v["measurements"]:
                fh.write(f"{m}\n")
        fh.write("```\n")
        fh.write(f"\n## Speed ({SPEED_CONFIG['n_rows']}x{SPEED_CONFIG['n_features']}, "
                 f"depth {DEPTH}, {ITERS} iters, RMSE)\n\n")
        fh.write("| bootstrap | rs on device? | catboost_rs s | CatBoost GPU s | "
                 "CatBoost CPU s | rs/CB-GPU |\n|---|---|---|---|---|---|\n")
        for r in rows:
            fh.write(f"| {r['bootstrap_type']} | "
                     f"{r['catboost_rs_device_resident_proven']} | "
                     f"{r['catboost_rs_s']} | {r['catboost_gpu_s']} | "
                     f"{r['catboost_cpu_s']} | {r['ratio_rs_over_catboost_gpu']} |\n")
        fh.write("\n## Caveats\n\n")
        for k, v in result["caveats"].items():
            fh.write(f"- **{k}**: {v}\n")
    log("\n" + open(os.path.join(WORK, "report.md")).read())
    log("DONE verdict=" + result["verdict"])


if __name__ == "__main__":
    main()
