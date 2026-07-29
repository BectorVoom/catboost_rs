#!/usr/bin/env python3
# catboost-rs bootstrap_type oracle + learning-speed sweep (Kaggle CUDA GPU).
#
# Answers one question end-to-end, on a real CUDA GPU, for EVERY bootstrap_type:
#   Part A (BLOCKING) — do the bootstrap parity oracles still pass?
#   Part B            — how fast does each bootstrap_type train, vs official CatBoost?
#
# HONESTY DISCIPLINE (bench/RESULTS.md house style). Two caveats are stated on every
# row and never dropped:
#
#   1. ONLY `bootstrap_type=No` CAN REACH THE GPU. `device_host_eligible` in
#      crates/cb-train/src/boosting.rs hard-requires
#      `matches!(params.bootstrap_type, EBootstrapType::No) && random_strength == 0.0`.
#      Every other bootstrap_type falls back to the CPU grower — the GPU is IDLE for
#      those rows. They are CPU numbers printed on a GPU machine, nothing more. The
#      device bootstrap kernels (`launch_bootstrap_weights_resident`,
#      `launch_mvs_weights_resident`) exist but are deliberately unwired (the WR-01
#      "NOT YET WIRED" comment at boosting.rs). This bench MEASURES that gap; it does
#      not paper over it.
#   2. Device activation is NOT observable from Python — there is no log line or
#      attribute proving the device path ran for a given fit. Even the `No` row cannot
#      be proven GPU-resident from this surface alone; Part A's cuda self-oracles are
#      the evidence that the device path works at all.
#
# Do-not-fabricate: correctness is gated BEFORE any timing; every number comes from a
# measured call in THIS run; a failed arm leaves `null`, never an invented value.
#
# Source is cloned from GitHub (kernel has enable_internet), NOT a Kaggle dataset, so
# the exact commit under test is recorded in result.json provenance.

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

WORK = "/kaggle/working"
REPO = "/tmp/repo"

GIT_URL = "https://github.com/BectorVoom/catboost_rs.git"
GIT_REF = "fix/bootstrap-rng-draw-accounting"

# Speed workload. Scaled to bound wall-clock while staying in the regime the earlier
# rounds measured (bench/quick_gpu_speed used the identical shape), so numbers are
# comparable across rounds.
SPEED_CONFIG = dict(n_rows=300_000, n_features=50, seed=42)
DEPTH = 6
ITERS = 30
LEARNING_RATE = 0.1
L2_LEAF_REG = 3.0
BORDER_COUNT = 32
RANDOM_SEED = 42

# The sweep. `subsample` / `bagging_temperature` mirror the oracle fixtures' pinning so
# each arm exercises a real sampler rather than a degenerate all-1.0 short-circuit.
# Poisson is included to RECORD upstream's CPU rejection, not as a pass condition.
BOOTSTRAP_ARMS = [
    ("No", {}),
    ("Bayesian", {"bagging_temperature": 1.0}),
    ("Bernoulli", {"subsample": 0.8}),
    ("MVS", {"subsample": 0.8}),
    ("Poisson", {"subsample": 0.8}),
]

# Part A oracle suites. (label, crate, cargo extra args, test filters, blocking?)
# The two cb-train suites are the CPU parity gate for this fix; the cb-backend cuda
# families are the device-side self-oracles for the (currently unwired) bootstrap
# kernels — they are what would have to stay green for WR-01 wiring to be credible.
ORACLE_SUITES = [
    ("cb-train bootstrap parity (CPU, all 4 types)", "cb-train",
     ["--test", "bootstrap_oracle_test"], [], True),
    ("cb-train regularization parity (CPU)", "cb-train",
     ["--test", "regularization_oracle_test"], [], False),
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
        """Run a command, stream-capture it, return (rc, tail)."""
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

    result["caveats"]["only_No_is_gpu_eligible"] = (
        "device_host_eligible (crates/cb-train/src/boosting.rs) hard-requires "
        "bootstrap_type == No AND random_strength == 0.0. Bayesian/Bernoulli/MVS/Poisson "
        "rows are CPU-grower numbers measured on a GPU machine; the GPU is idle for them."
    )
    result["caveats"]["device_activation_not_observable"] = (
        "No Python-visible signal proves the device grow path ran for a given fit; a "
        "silent CPU fallback cannot be ruled out from this surface."
    )

    # -----------------------------------------------------------------
    # STEP 1 — provenance
    # -----------------------------------------------------------------
    rc, out = sh("nvidia-smi --query-gpu=name,driver_version,memory.total "
                 "--format=csv,noheader")
    result["provenance"]["gpu"] = (out or "").strip().splitlines()[0] if rc == 0 and out.strip() else None
    if not result["provenance"]["gpu"]:
        result["fatal"] = "no GPU visible to nvidia-smi — refusing to quote GPU numbers"
        json.dump(result, open(os.path.join(WORK, "result.json"), "w"), indent=2)
        log("FATAL:", result["fatal"])
        sys.exit(2)
    log("GPU:", result["provenance"]["gpu"])
    sh("nvcc --version || true")
    result["provenance"]["cpu_count"] = os.cpu_count()

    # -----------------------------------------------------------------
    # STEP 2 — stage source from GitHub
    # -----------------------------------------------------------------
    if os.path.exists(REPO):
        shutil.rmtree(REPO)
    rc, _ = sh(["git", "clone", "--depth", "1", "--branch", GIT_REF, GIT_URL, REPO],
               timeout=1800)
    if rc != 0 or not os.path.exists(os.path.join(REPO, "Cargo.toml")):
        result["fatal"] = f"git clone of {GIT_REF} failed"
        json.dump(result, open(os.path.join(WORK, "result.json"), "w"), indent=2)
        log("FATAL:", result["fatal"])
        sys.exit(2)
    _, sha = sh(["git", "-C", REPO, "rev-parse", "HEAD"])
    result["provenance"]["git_ref"] = GIT_REF
    result["provenance"]["git_sha"] = (sha or "").strip()

    # Provenance marker: prove the staged tree really carries THIS fix, so a stale
    # clone can never masquerade as a verified run.
    rc_m, out_m = sh("grep -c 'POST_TREE_EXTRA_DRAWS: usize = 2' "
                     f"{REPO}/crates/cb-train/src/boosting.rs || true")
    digits = [ln.strip() for ln in (out_m or "").splitlines() if ln.strip().isdigit()]
    result["provenance"]["staged_source_has_rng_fix"] = bool(digits and int(digits[0]) > 0)
    log("staged_source_has_rng_fix:", result["provenance"]["staged_source_has_rng_fix"])

    # -----------------------------------------------------------------
    # STEP 3 — rust toolchain
    # -----------------------------------------------------------------
    sh("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
       "sh -s -- -y --default-toolchain stable --profile minimal", timeout=1800)
    env = os.environ.copy()
    env["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + env["PATH"]
    env["CARGO_TARGET_DIR"] = "/tmp/target"
    env["CARGO_BUILD_JOBS"] = "2"
    env["CARGO_NET_RETRY"] = "5"
    env["RUST_BACKTRACE"] = "1"
    rc, out = sh("rustc --version && cargo --version", env=env)
    result["provenance"]["rust"] = (out or "").strip()

    # =================================================================
    # PART A — ORACLE GATE (blocking on the CPU bootstrap parity suite)
    # =================================================================
    log("\n" + "=" * 70 + "\nPART A — oracle\n" + "=" * 70)
    for label, crate, extra, filters, blocking in ORACLE_SUITES:
        cmd = (["cargo", "test", "--release", "-p", crate] + extra + filters
               + ["--", "--include-ignored", "--test-threads", "1"])
        rc, out = sh(cmd, cwd=REPO, env=env, timeout=7200)
        # Parse every "test result:" line the invocation emitted. Regex, NOT
        # positional split — the counts sit mid-line ("test result: ok. 5 passed; ...")
        # so token-index parsing reads the wrong field on the first segment.
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
        result["oracle"][label] = {
            "rc": rc, "passed": passed, "failed": failed, "ignored": ignored,
            "blocking": blocking, "tail": (out or "")[-3000:],
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

    # =================================================================
    # PART B — SPEED SWEEP
    # =================================================================
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

    wheels = sorted(glob.glob("/tmp/target/wheels/*.whl"), key=os.path.getmtime)
    if not wheels:
        result["verdict"] = "ORACLE-PASS/BUILD-FAIL"
        result["fatal"] = "no .whl produced"
        json.dump(result, open(os.path.join(WORK, "result.json"), "w"), indent=2)
        sys.exit(2)
    sh(["pip", "install", "--force-reinstall", wheels[-1]], env=env, timeout=1800)

    import catboost_rs
    try:
        import catboost
    except Exception:
        sh("pip install -q catboost", env=env, timeout=1800)
        import catboost
    result["speed"]["catboost_version"] = getattr(catboost, "__version__", "unknown")

    sys.path.insert(0, os.path.join(REPO, "bench"))
    import generator
    import numpy as np

    X, y = generator.generate(**SPEED_CONFIG)
    result["speed"]["config"] = {
        "speed_config": SPEED_CONFIG, "depth": DEPTH, "iters": ITERS,
        "learning_rate": LEARNING_RATE, "l2_leaf_reg": L2_LEAF_REG,
        "border_count": BORDER_COUNT, "random_seed": RANDOM_SEED,
        "loss": "RMSE", "X_shape": list(X.shape),
    }

    def timed_fit(arm, make_model):
        """Warm (untimed, absorbs JIT) then timed fit; predict() drains the lazy
        CubeCL queue before the clock stops. Returns (seconds, train_rmse) or
        (None, None) with the error recorded."""
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
        gpu_eligible = (bt == "No")

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

        log(f"\n--- bootstrap_type={bt} (catboost_rs GPU-eligible: {gpu_eligible}) ---")
        rs_s, rs_q = timed_fit(f"catboost_rs[{bt}]", rs_model)
        cbg_s, cbg_q = timed_fit(f"catboost_gpu[{bt}]", cb_gpu)
        cbc_s, cbc_q = timed_fit(f"catboost_cpu[{bt}]", cb_cpu)

        rows.append({
            "bootstrap_type": bt,
            "catboost_rs_uses_gpu": gpu_eligible,
            "catboost_rs_s": rs_s, "catboost_rs_train_rmse": rs_q,
            "catboost_gpu_s": cbg_s, "catboost_gpu_train_rmse": cbg_q,
            "catboost_cpu_s": cbc_s, "catboost_cpu_train_rmse": cbc_q,
            "ratio_rs_over_cb_gpu": (round(rs_s / cbg_s, 3)
                                     if rs_s and cbg_s else None),
            "ratio_rs_over_cb_cpu": (round(rs_s / cbc_s, 3)
                                     if rs_s and cbc_s else None),
        })
        json.dump(result | {"speed": result["speed"] | {"rows": rows}},
                  open(os.path.join(WORK, "result.json"), "w"), indent=2)

    result["speed"]["rows"] = rows

    # -----------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------
    with open(os.path.join(WORK, "report.md"), "w") as fh:
        fh.write("# catboost-rs — bootstrap_type oracle + learning speed (CUDA GPU)\n\n")
        fh.write(f"- GPU: `{result['provenance']['gpu']}`\n")
        fh.write(f"- Commit: `{result['provenance']['git_sha']}` (`{GIT_REF}`)\n")
        fh.write(f"- RNG-fix marker present: **{result['provenance']['staged_source_has_rng_fix']}**\n")
        fh.write(f"- Verdict: **{result['verdict']}**\n\n")

        fh.write("## Part A — oracle\n\n")
        fh.write("| suite | rc | passed | failed | blocking |\n|---|---|---|---|---|\n")
        for label, d in result["oracle"].items():
            fh.write(f"| {label} | {d['rc']} | {d['passed']} | {d['failed']} | "
                     f"{d['blocking']} |\n")

        fh.write("\n## Part B — learning speed by bootstrap_type "
                 f"({SPEED_CONFIG['n_rows']}x{SPEED_CONFIG['n_features']}, "
                 f"depth {DEPTH}, {ITERS} iters, RMSE)\n\n")
        fh.write("| bootstrap_type | rs on GPU? | catboost_rs (s) | CatBoost GPU (s) | "
                 "CatBoost CPU (s) | rs/cbGPU | rs train RMSE |\n")
        fh.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            fh.write(f"| {r['bootstrap_type']} | {r['catboost_rs_uses_gpu']} | "
                     f"{r['catboost_rs_s']} | {r['catboost_gpu_s']} | "
                     f"{r['catboost_cpu_s']} | {r['ratio_rs_over_cb_gpu']} | "
                     f"{r['catboost_rs_train_rmse']} |\n")

        fh.write("\n## Caveats (never dropped)\n\n")
        for k, v in result["caveats"].items():
            fh.write(f"- **{k}**: {v}\n")

    json.dump(result, open(os.path.join(WORK, "result.json"), "w"), indent=2)
    log("\nDONE — verdict:", result["verdict"])


if __name__ == "__main__":
    main()
