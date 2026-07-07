#!/usr/bin/env python3
# catboost-rs quick GPU-vs-GPU speed check (Kaggle CUDA) — quick task 260707-rr0.
#
# One-off, INFORMAL benchmark (NOT the formal Phase-22 DX-01 benchmark). It compares
# training wall-clock of official CatBoost (task_type='GPU') against catboost_rs — this
# repo's Rust reimplementation, built as a Python wheel with the `cuda` Cargo feature —
# through catboost_rs's real public Python .fit() API (NOT a Rust cargo-test oracle).
#
# Honesty discipline (bench/RESULTS.md house style): no speed number is trusted until the
# wheel built+installed cleanly and nvidia-smi shows a real GPU. Every number in
# result.json / report.md comes from an actual measured call here — nothing is hardcoded.
#
# IMPORTANT CAVEAT (device activation is NOT observable from Python): catboost_rs's GPU
# tree-growth loop only activates when the per-fit `device_host_eligible` gate holds
# (crates/cb-train/src/boosting.rs). This bench satisfies every known precondition BY
# CONSTRUCTION (see build_eligibility_audit()), but there is NO log line / public
# attribute to prove the device path actually ran for a given fit. A silent CPU fallback
# therefore cannot be 100% ruled out from the Python surface — the report states this
# plainly and never drops the caveat, even if the numbers look device-fast.
#
# All run-time work lives inside main() under `if __name__ == "__main__"`, so importing
# this module in-env (no GPU, no numpy) exposes the helpers and triggers no run.

WORK = "/kaggle/working"

# Workload + model config (module constants, no side effects). Scaled DOWN from the
# canonical SPEED_CONFIG (1e6x50) to 300k rows to bound the kernel wall-clock while still
# reusing bench/generator.py's seeded generate()/binary_target().
SPEED_CONFIG = dict(n_rows=300_000, n_features=50, seed=42)
DEPTH = 6
ITERS = 30
LEARNING_RATE = 0.1
L2_LEAF_REG = 3.0
BORDER_COUNT = 32
RANDOM_SEED = 42


def build_eligibility_audit():
    """Static, no-instrumentation audit (fact 10) of catboost_rs's device_host_eligible
    preconditions that THIS bench's config satisfies by construction.

    Returns a dict: each known precondition -> {"satisfied": True, "rationale": <str>},
    plus a top-level "activation_observable": False and a "caveat" string disclosing that
    device activation itself is not instrumented/observable from the Python surface, so a
    silent CPU fallback cannot be 100% ruled out in this informal check.
    """
    conds = {
        "grow_policy_symmetric": {
            "satisfied": True,
            "rationale": "default grow_policy (SymmetricTree/oblivious) — no grow_policy kwarg passed.",
        },
        "single_dim_target": {
            "satisfied": True,
            "rationale": "approx_dimension == 1 — single-dim RMSE regression / binary Logloss, not multiclass/multilabel.",
        },
        "bootstrap_type_no": {
            "satisfied": True,
            "rationale": "bootstrap_type='No' passed explicitly (also the Python default).",
        },
        "random_strength_zero": {
            "satisfied": True,
            "rationale": "random_strength=0.0 passed explicitly (also the default).",
        },
        "leaf_estimation_gradient": {
            "satisfied": True,
            "rationale": "leaf_estimation_method='Gradient' passed explicitly (Gradient/Simple are eligible).",
        },
        "unit_weights": {
            "satisfied": True,
            "rationale": "no sample_weight passed -> unit object weights.",
        },
        "boost_from_average_false": {
            "satisfied": True,
            "rationale": "boost_from_average=False passed explicitly (bias==0.0; the default True silently falls back to the CPU grower for RMSE).",
        },
        "no_eval_set": {
            "satisfied": True,
            "rationale": "no eval_set passed to fit().",
        },
        "no_monotone_constraints": {
            "satisfied": True,
            "rationale": "no monotone constraints configured.",
        },
        "pure_float_features": {
            "satisfied": True,
            "rationale": "X is pure float32 — no categorical / text / embedding columns.",
        },
        "no_ranking_groups": {
            "satisfied": True,
            "rationale": "single fold, no ranking group ids — plain regression/classification.",
        },
    }
    return {
        "preconditions": conds,
        "activation_observable": False,
        "caveat": (
            "Device activation is NOT directly instrumented or observable from the Python "
            "surface in this informal check: catboost_rs exposes no log line or public "
            "attribute indicating whether the GPU tree-growth loop actually ran for a given "
            ".fit(). This bench satisfies every known device_host_eligible precondition BY "
            "CONSTRUCTION (see the preconditions map above), but that is a static/documented "
            "audit, not a runtime proof. A silent CPU fallback therefore cannot be 100% ruled "
            "out from the Python surface alone. If a catboost_rs timing lands in the same "
            "ballpark as a known host-CPU reference rather than a device-fast number, treat it "
            "as a possible silent CPU fallback."
        ),
    }


def main():
    import os, subprocess, sys, shutil, time, json, glob

    os.makedirs(WORK, exist_ok=True)

    def log(*a):
        print(*a, flush=True)

    def sh(cmd, cwd=None, env=None, timeout=None):
        log("\n$", cmd if isinstance(cmd, str) else " ".join(cmd))
        try:
            r = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, env=env,
                               shell=isinstance(cmd, str), timeout=timeout)
        except subprocess.TimeoutExpired as e:
            log("TIMEOUT after", timeout, "s")
            return 124, "TIMEOUT: " + str(e)
        out = (r.stdout or "") + (("\nSTDERR:\n" + r.stderr) if r.stderr else "")
        log(out[-6000:])
        return r.returncode, out

    def env_line(cmd):
        try:
            return subprocess.run(cmd, text=True, capture_output=True, shell=True).stdout.strip()
        except Exception as e:
            return f"(err {e})"

    result = {"kind": "quick-gpu-speed-check", "task": "260707-rr0"}

    # ---------------------------------------------------------------
    # STEP 1 — environment provenance
    # ---------------------------------------------------------------
    log("=================== ENVIRONMENT ===================")
    sh("nvidia-smi", timeout=120)
    result["provenance"] = {
        "gpu": env_line("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"),
        "driver": env_line("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
        "nvcc": env_line("nvcc --version 2>/dev/null | grep -oE 'release [0-9.]+' | head -1 || echo NO_NVCC"),
        "cuda_dirs": env_line("ls -d /usr/local/cuda* 2>/dev/null | tr '\\n' ' '"),
    }
    log("provenance:", result["provenance"])

    # ---------------------------------------------------------------
    # STEP 2 — stage the repo into /tmp
    # ---------------------------------------------------------------
    inp = "/kaggle/input"
    tarball = srcdir = None
    for dp, _, fs in os.walk(inp):
        if "repo.tar.gz" in fs:
            tarball = os.path.join(dp, "repo.tar.gz")
        if "Cargo.toml" in fs and os.path.exists(os.path.join(dp, "crates")):
            srcdir = dp
    REPO = "/tmp/repo"
    if os.path.exists(REPO):
        shutil.rmtree(REPO)
    os.makedirs(REPO)
    if tarball:
        sh(["tar", "-xzf", tarball, "-C", REPO])
    elif srcdir:
        sh(["bash", "-lc", f"cp -a '{srcdir}/.' '{REPO}/'"])
    else:
        result["fatal"] = "payload not found under /kaggle/input"
        json.dump(result, open(os.path.join(WORK, "result.json"), "w"), indent=2)
        log("FATAL: no source payload found")
        sys.exit(2)
    if not os.path.exists(os.path.join(REPO, "Cargo.toml")) or \
       not os.path.exists(os.path.join(REPO, "crates/catboost-rs-py/Cargo.toml")):
        result["fatal"] = "Cargo.toml or crates/catboost-rs-py/Cargo.toml missing after stage"
        json.dump(result, open(os.path.join(WORK, "result.json"), "w"), indent=2)
        log("FATAL:", result["fatal"])
        sys.exit(2)
    log("REPO staged ->", sorted(os.listdir(REPO))[:20])

    # ---------------------------------------------------------------
    # STEP 3 — Rust toolchain
    # ---------------------------------------------------------------
    sh("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
       "sh -s -- -y --default-toolchain stable --profile minimal")
    env = os.environ.copy()
    env["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + env["PATH"]
    env["CARGO_TARGET_DIR"] = "/tmp/target"
    env["CARGO_BUILD_JOBS"] = "2"
    env["CARGO_NET_RETRY"] = "5"
    env["RUST_BACKTRACE"] = "1"
    sh("rustc --version && cargo --version", env=env)

    # ---------------------------------------------------------------
    # STEP 4 — build the cuda wheel (fact 2)
    # ---------------------------------------------------------------
    sh("pip install -q maturin", env=env, timeout=1200)
    rc, out = sh(
        ["maturin", "build", "--release", "--no-default-features", "--features", "cuda",
         "-m", "/tmp/repo/crates/catboost-rs-py/Cargo.toml"],
        cwd=REPO, env=env, timeout=3600,
    )
    result["build_ok"] = (rc == 0)
    result["build_tail"] = out[-4000:]
    if not result["build_ok"]:
        result["import_ok"] = False
        json.dump(result, open(os.path.join(WORK, "result.json"), "w"), indent=2)
        with open(os.path.join(WORK, "report.md"), "w") as fh:
            fh.write("# catboost-rs quick GPU speed check — BUILD FAILED\n\n")
            fh.write(f"GPU: {result['provenance']['gpu']}\n\n")
            fh.write("The cuda-feature wheel failed to build; no speed number is reported "
                     "(do-not-fabricate).\n\n## Build tail\n\n```\n")
            fh.write(result["build_tail"])
            fh.write("\n```\n")
        log("FATAL: wheel build failed rc=", rc)
        sys.exit(2)

    # ---------------------------------------------------------------
    # STEP 5 — install + import
    # ---------------------------------------------------------------
    wheels = sorted(glob.glob("/tmp/target/wheels/*.whl"), key=os.path.getmtime)
    result["wheel"] = wheels[-1] if wheels else None
    if not wheels:
        result["fatal"] = "no .whl produced under /tmp/target/wheels"
        json.dump(result, open(os.path.join(WORK, "result.json"), "w"), indent=2)
        log("FATAL:", result["fatal"])
        sys.exit(2)
    sh(["pip", "install", "--force-reinstall", wheels[-1]], env=env, timeout=1200)
    try:
        import catboost_rs
        result["import_ok"] = (hasattr(catboost_rs, "CatBoostRegressor")
                               and hasattr(catboost_rs, "CatBoostClassifier"))
    except Exception as e:
        result["import_ok"] = False
        result["import_error"] = repr(e)
        json.dump(result, open(os.path.join(WORK, "result.json"), "w"), indent=2)
        log("FATAL: import catboost_rs failed:", e)
        sys.exit(2)
    try:
        import catboost  # noqa: F401
    except Exception:
        sh("pip install -q catboost", env=env, timeout=1800)
        import catboost  # noqa: F401
    result["catboost_version"] = getattr(catboost, "__version__", "unknown")
    log("import_ok:", result["import_ok"], "catboost:", result["catboost_version"])

    # ---------------------------------------------------------------
    # STEP 6 — workload (reuse bench/generator.py, fact 6)
    # ---------------------------------------------------------------
    sys.path.insert(0, os.path.join(REPO, "bench"))
    import generator
    X, y_reg = generator.generate(**SPEED_CONFIG)
    y_bin = generator.binary_target(X, SPEED_CONFIG["seed"])
    result["config"] = {
        "speed_config": SPEED_CONFIG, "depth": DEPTH, "iters": ITERS,
        "learning_rate": LEARNING_RATE, "l2_leaf_reg": L2_LEAF_REG,
        "border_count": BORDER_COUNT, "random_seed": RANDOM_SEED,
        "X_shape": list(X.shape),
    }

    # ---------------------------------------------------------------
    # STEP 7 — device-eligibility precondition audit (static, no instrumentation)
    # ---------------------------------------------------------------
    result["device_eligibility_preconditions"] = build_eligibility_audit()

    # ---------------------------------------------------------------
    # STEP 8 — warm/drain timed fit helper (fact 9)
    # ---------------------------------------------------------------
    result["timings"] = {"errors": {}}

    def timed_fit(arm_name, make_model, Xd, yd):
        try:
            warm = make_model()
            warm.fit(Xd, yd)                  # UNTIMED warm/JIT-absorbing run
        except Exception as e:
            log(f"[{arm_name}] warm fit FAILED: {e}")
            result["timings"]["errors"][arm_name] = "warm: " + repr(e)
            return None
        try:
            model = make_model()
            t0 = time.time()
            model.fit(Xd, yd)                 # timed run
            _ = model.predict(Xd[:1024])      # drain lazy CubeCL queue before stopping clock
            _ = list(_[:1]) if hasattr(_, "__iter__") else _
            elapsed = round(time.time() - t0, 4)
            log(f"[{arm_name}] fit_s={elapsed}")
            return elapsed
        except Exception as e:
            log(f"[{arm_name}] timed fit FAILED: {e}")
            result["timings"]["errors"][arm_name] = "timed: " + repr(e)
            return None

    # ---------------------------------------------------------------
    # STEP 9 — run all 4 arms
    # ---------------------------------------------------------------
    def rs_rmse():
        return catboost_rs.CatBoostRegressor(
            iterations=ITERS, depth=DEPTH, learning_rate=LEARNING_RATE,
            l2_leaf_reg=L2_LEAF_REG, border_count=BORDER_COUNT, loss_function="RMSE",
            bootstrap_type="No", random_strength=0.0,
            leaf_estimation_method="Gradient", boost_from_average=False)

    def rs_logloss():
        return catboost_rs.CatBoostClassifier(
            iterations=ITERS, depth=DEPTH, learning_rate=LEARNING_RATE,
            l2_leaf_reg=L2_LEAF_REG, border_count=BORDER_COUNT, loss_function="Logloss",
            bootstrap_type="No", random_strength=0.0,
            leaf_estimation_method="Gradient", boost_from_average=False)

    def cb_rmse():
        return catboost.CatBoostRegressor(
            iterations=ITERS, depth=DEPTH, learning_rate=LEARNING_RATE,
            l2_leaf_reg=L2_LEAF_REG, border_count=BORDER_COUNT, loss_function="RMSE",
            task_type="GPU", devices="0", bootstrap_type="No", random_strength=0.0,
            boost_from_average=False, random_seed=RANDOM_SEED, verbose=False)

    def cb_logloss():
        return catboost.CatBoostClassifier(
            iterations=ITERS, depth=DEPTH, learning_rate=LEARNING_RATE,
            l2_leaf_reg=L2_LEAF_REG, border_count=BORDER_COUNT, loss_function="Logloss",
            task_type="GPU", devices="0", bootstrap_type="No", random_strength=0.0,
            boost_from_average=False, random_seed=RANDOM_SEED, verbose=False)

    t = result["timings"]
    t["catboost_rs_rmse_s"] = timed_fit("catboost_rs_rmse", rs_rmse, X, y_reg)
    t["catboost_rs_logloss_s"] = timed_fit("catboost_rs_logloss", rs_logloss, X, y_bin)
    t["catboost_official_gpu_rmse_s"] = timed_fit("catboost_official_gpu_rmse", cb_rmse, X, y_reg)
    t["catboost_official_gpu_logloss_s"] = timed_fit("catboost_official_gpu_logloss", cb_logloss, X, y_bin)

    def ratio(official, rs):
        if isinstance(official, (int, float)) and isinstance(rs, (int, float)) and rs > 0:
            return round(official / rs, 4)
        return None

    result["speedup"] = {
        # ratio > 1 => catboost_rs faster; < 1 => official CatBoost GPU faster.
        "rmse_official_over_rs": ratio(t["catboost_official_gpu_rmse_s"], t["catboost_rs_rmse_s"]),
        "logloss_official_over_rs": ratio(t["catboost_official_gpu_logloss_s"], t["catboost_rs_logloss_s"]),
    }

    # ---------------------------------------------------------------
    # STEP 10 — emit output
    # ---------------------------------------------------------------
    result["date"] = time.strftime("%Y-%m-%d", time.gmtime())
    json.dump(result, open(os.path.join(WORK, "result.json"), "w"), indent=2)

    def fmt(v):
        return "N/A" if v is None else f"{v:.4f}"

    def fmtx(v):
        return "N/A" if v is None else f"{v:.4f}x"

    caveat = result["device_eligibility_preconditions"]["caveat"]
    with open(os.path.join(WORK, "report.md"), "w") as fh:
        fh.write("# catboost-rs quick GPU-vs-GPU speed check (informal)\n\n")
        fh.write(f"GPU: {result['provenance']['gpu']}  \n")
        fh.write(f"driver: {result['provenance']['driver']}  nvcc: {result['provenance']['nvcc']}  \n")
        fh.write(f"build_ok: {result['build_ok']}  import_ok: {result['import_ok']}  "
                 f"catboost: {result['catboost_version']}\n\n")
        fh.write("## Wall-clock (300k x 50, depth 6, 30 iters)\n\n")
        fh.write("| arm | seconds |\n|---|---|\n")
        fh.write(f"| catboost_rs RMSE | {fmt(t['catboost_rs_rmse_s'])} |\n")
        fh.write(f"| catboost_rs Logloss | {fmt(t['catboost_rs_logloss_s'])} |\n")
        fh.write(f"| official CatBoost GPU RMSE | {fmt(t['catboost_official_gpu_rmse_s'])} |\n")
        fh.write(f"| official CatBoost GPU Logloss | {fmt(t['catboost_official_gpu_logloss_s'])} |\n\n")
        fh.write("## Speedup (official / catboost_rs; >1 => catboost_rs faster)\n\n")
        fh.write("| loss | official_over_rs |\n|---|---|\n")
        fh.write(f"| RMSE | {fmtx(result['speedup']['rmse_official_over_rs'])} |\n")
        fh.write(f"| Logloss | {fmtx(result['speedup']['logloss_official_over_rs'])} |\n\n")
        if t["errors"]:
            fh.write("## Arm errors\n\n")
            for k, v in t["errors"].items():
                fh.write(f"- **{k}**: {v}\n")
            fh.write("\n")
        fh.write("## Device-activation caveat (always included)\n\n")
        fh.write(caveat + "\n")

    log("\n=================== SUMMARY ===================")
    log("gpu:", result["provenance"]["gpu"])
    log("build_ok:", result["build_ok"], "import_ok:", result["import_ok"],
        "catboost:", result["catboost_version"])
    log("catboost_rs_rmse_s:", t["catboost_rs_rmse_s"])
    log("catboost_rs_logloss_s:", t["catboost_rs_logloss_s"])
    log("catboost_official_gpu_rmse_s:", t["catboost_official_gpu_rmse_s"])
    log("catboost_official_gpu_logloss_s:", t["catboost_official_gpu_logloss_s"])
    log("speedup rmse_official_over_rs:", result["speedup"]["rmse_official_over_rs"])
    log("speedup logloss_official_over_rs:", result["speedup"]["logloss_official_over_rs"])
    log("CAVEAT:", caveat)
    log("=================== QUICK GPU SPEED CHECK COMPLETE ===================")


if __name__ == "__main__":
    main()
